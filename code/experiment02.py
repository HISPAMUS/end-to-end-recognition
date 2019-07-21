import argparse
from data import DataReader, prepare_data
from model import default_model_params, cnn_block, rnn_block, sparse_tensor_to_strs, sparse_tuple_from
import sys
import tensorflow as tf
from utils import config, eval, get_logger, ResultsManager


# ===================================================


def model(params):
    placeholders = {}

    input = tf.placeholder(shape=(None,
                                  params['img_height'],
                                  params['img_width'],
                                  params['img_channels']),  # [batch, height, width, channels]
                           dtype=tf.float32,
                           name='model_input')

    seq_len = tf.placeholder(tf.int32, [None], name='seq_lengths') # Real length of the image    
    rnn_keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    
    placeholders['input'] = input
    placeholders['seq_len'] = seq_len
    placeholders['keep_prob'] = rnn_keep_prob

    input_shape = tf.shape(input)

    with tf.variable_scope('symbol'):
        with tf.variable_scope('cnn'):
            # Convolutional blocks
            x, width_reduction, height_reduction = cnn_block(input, params)

            # Prepare output of conv block for recurrent blocks
            features = tf.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
            feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
            feature_width = input_shape[2] // width_reduction
            features = tf.reshape(features, tf.stack([tf.cast(feature_width, 'int32'), input_shape[0],
                                                    tf.cast(feature_dim, 'int32')]))  # -> [width, batch, features]

            # Used for prediction
            #tf.constant(params['img_height'], name='input_height')
            #tf.constant(width_reduction, name='width_reduction')

        with tf.variable_scope('rnn'):
            placeholders['symbol'] = rnn_block(features, placeholders, params, 0)

    with tf.variable_scope('position'):
        with tf.variable_scope('cnn'):
            # Convolutional blocks
            x, width_reduction, height_reduction = cnn_block(input, params)

            # Prepare output of conv block for recurrent blocks
            features = tf.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
            feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
            feature_width = input_shape[2] // width_reduction
            features = tf.reshape(features, tf.stack([tf.cast(feature_width, 'int32'), input_shape[0],
                                                    tf.cast(feature_dim, 'int32')]))  # -> [width, batch, features]

            # Used for prediction
            #tf.constant(params['img_height'], name='input_height')
            #tf.constant(width_reduction, name='width_reduction')

        with tf.variable_scope('rnn'):
            placeholders['position'] = rnn_block(features, placeholders, params, 1)
    
    with tf.variable_scope('joint'):
        rnn_outputs = (placeholders['symbol']['rnn_outputs'], placeholders['position']['rnn_outputs'])
        rnn_outputs = tf.concat(rnn_outputs, 2)

        logits = tf.layers.dense(rnn_outputs, params['vocabulary_sizes'][2]+1) # +1 because of 'blank' CTC

        # Add softmax!
        softmax = tf.nn.softmax(logits)
        
        #tf.add_to_collection("softmax", softmax) # for restoring purposes
        #tf.add_to_collection("logits", logits)  # for restoring purposes

        # CTC Loss computation
        targets = tf.sparse_placeholder(dtype=tf.int32, name='target')
        ctc_loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=placeholders['seq_len'], time_major=True)
        loss = tf.reduce_mean(ctc_loss)
        
        placeholders['joint'] = {}
        placeholders['joint']['target'] = targets
        placeholders['joint']['logits'] = logits
        placeholders['joint']['softmax'] = softmax
        placeholders['joint']['loss'] = loss


    return placeholders
      

# ===================================================


if __name__ == '__main__':

    experiment = "exp02"

    parser = argparse.ArgumentParser(description='CRNN Training for HMR.')
    
    # DataReader options
    parser.add_argument('--input-data', dest='data_path', type=str, required=True, help='Path to data file')
    parser.add_argument('--image-height', dest='image_height', type=int, default=64, help='Image size will be reduced to this height')
    parser.add_argument('--channels', dest='channels', type=int, default=1, help='Number of channels in training')
    parser.add_argument('--image-transformations', dest='image_transformations', type=int, default=1, help='Data augmentation: number or transformations to apply to the images in the training set. Value 1 (default) disables data augmentation')
    parser.add_argument('--sequence-delimiter', dest='sequence_delimiter', default=False, action='store_true', help='Use or not sequence delimiters <s> (start) and <e> (end)')
    parser.add_argument('--test-split', dest='test_split', type=float, default=0.1, help='% of samples for testing')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16, help='Batch size')

    # Training options
    parser.add_argument('--split-epochs', dest='split_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--joint-epochs', dest='joint_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--gpu', dest='gpu', type=str, default=None, help='GPU id')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='Random seed for shuffling data (default=None)')
    parser.add_argument('--model-symbol', dest='model_symbol', type=str, default=None, help='Load symbol model from file')
    parser.add_argument('--model-position', dest='model_position', type=str, default=None, help='Load position model from file')
    parser.add_argument('--skip-split-training', dest='skip_split_training', default=False, action='store_true', help='Requires symbol and position models')
    parser.add_argument('--freeze', dest='freeze', type=int, default=0, help='Freeze point (0=No freeze, 1=Freeze convolution layer, 2=Freeze conv+rnn')

    FLAGS = parser.parse_args()

    if FLAGS.skip_split_training and (FLAGS.model_symbol is None or FLAGS.model_position is None):
        print('Symbol and position models are required in order to skip split training')
        sys.exit()

    # ===============================================
    # Initialize logger & results manager
    logger = get_logger(experiment, FLAGS)
    results = ResultsManager(logger)

    # ===============================================
    # Initialize TensorFlow
    sess = config(FLAGS)

    # ===============================================
    # Loading data
    print('Preparing data...')
    
    data_reader = DataReader(
        FLAGS.data_path,
        image_height=FLAGS.image_height,
        channels=FLAGS.channels,
        sequence_delimiter=FLAGS.sequence_delimiter,
        test_split=FLAGS.test_split,
        batch_size=FLAGS.batch_size,
        image_transformations=FLAGS.image_transformations,
        seed=FLAGS.seed
    )
    
    train_ds, val_ds, test_ds = data_reader.get_data()
    vocabularies = data_reader.get_dictionaries() # 0 -> symbol, 1 -> position, 2 -> joint
    vocabulary_sizes = (
        len(vocabularies[0].word2idx),
        len(vocabularies[1].word2idx),
        len(vocabularies[2].word2idx)
    )

    print('Done')

    # ===============================================
    # Setting params
    params = default_model_params(FLAGS.image_height, FLAGS.channels, vocabulary_sizes, FLAGS.batch_size)

    # ===============================================
    # CRNN
    print("Creating model...")
       
    model_placeholders = model(params)

    optimizer_symbol = tf.train.AdamOptimizer().minimize(model_placeholders['symbol']['loss'])
    decoder_symbol, log_prob_symbol = tf.nn.ctc_greedy_decoder(model_placeholders['symbol']['logits'], model_placeholders['seq_len'])

    optimizer_position = tf.train.AdamOptimizer().minimize(model_placeholders['position']['loss'])
    decoder_position, log_prob_position = tf.nn.ctc_greedy_decoder(model_placeholders['position']['logits'], model_placeholders['seq_len'])

    if FLAGS.freeze == 1:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='symbol/rnn') + \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='position/rnn') + \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='joint')
    elif FLAGS.freeze == 2:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='joint')
    else:
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    optimizer_joint = tf.train.AdamOptimizer().minimize(model_placeholders['joint']['loss'], var_list=var_list)
    decoder_joint, log_prob_joint = tf.nn.ctc_greedy_decoder(model_placeholders['joint']['logits'], model_placeholders['seq_len'])

    print("Done")

    # ===============================================
    # Training   
    print('Training with ' + str(data_reader.TRAIN_SPLIT) + ' samples.')
    print('Validating with ' + str(data_reader.VAL_SPLIT) + ' samples.')
    print('Testing with ' + str(data_reader.TEST_SPLIT) + ' samples.')

    symbol_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='symbol')
    position_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='position')
    saver_symbol = tf.train.Saver(var_list=symbol_variables, max_to_keep=1)
    saver_position = tf.train.Saver(var_list=position_variables, max_to_keep=1)
    saver_joint = tf.train.Saver(max_to_keep=1) # Saves the complete model

    sess.run(tf.global_variables_initializer())

    # ===============================================
    # Load models
    if FLAGS.model_symbol is not None:
        saver_symbol.restore(sess, FLAGS.model_symbol)
        print('Loaded symbol model')

    if FLAGS.model_position is not None:
        saver_position.restore(sess, FLAGS.model_position)
        print('Loaded position model')

    # ===============================================
    # Split training
    if not FLAGS.skip_split_training:
        for epoch in range(1, FLAGS.split_epochs+1):
            print("Split epoch {}/{}".format(epoch, FLAGS.split_epochs))

            it_train = train_ds.make_one_shot_iterator()
            next_batch = it_train.get_next()
            batch = 1
            while True:
                try:
                    X, XL, Y_symbol, Y_position, Y_joint, YL = prepare_data(
                        sess.run(next_batch),
                        vocabulary_sizes,
                        params
                    )

                    print('Batch {}: {} samples'.format(batch, len(X)))

                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops): # Enables batch normalization
                        _, _ = sess.run(
                            [optimizer_symbol, optimizer_position], 
                            {
                                model_placeholders['input']: X,
                                model_placeholders['seq_len']: XL,
                                model_placeholders['symbol']['target']: sparse_tuple_from(Y_symbol),
                                model_placeholders['position']['target']: sparse_tuple_from(Y_position),
                                model_placeholders['keep_prob']: 0.75
                            }
                        )

                    batch = batch + 1
                except tf.errors.OutOfRangeError:
                    break
            
            # ===============================================
            # Split validation   
            if epoch % 1 == 0:
                metrics_symbol = (0, 0, 0) # (editions, total_length, sequences)
                metrics_position = (0, 0, 0)

                it_val = val_ds.make_one_shot_iterator()
                next_batch = it_val.get_next()
                while True:
                    try:
                        X, XL, Y_symbol, Y_position, Y_joint, YL = prepare_data(
                            sess.run(next_batch),
                            vocabulary_sizes,
                            params
                        )

                        pred_symbol, pred_position = sess.run(
                            [decoder_symbol, decoder_position],
                            {
                                model_placeholders['input']: X,
                                model_placeholders['seq_len']: XL,
                                model_placeholders['keep_prob']: 1.0,
                            }
                        )

                        metrics_symbol, H, Y = eval(pred_symbol, Y_symbol, vocabularies[0], metrics_symbol)
                        logger.log_predictions(epoch, H, Y)

                        metrics_position, H, Y = eval(pred_position, Y_position, vocabularies[1], metrics_position)
                        logger.log_predictions(epoch, H, Y)

                    except tf.errors.OutOfRangeError:
                        break

                save_symbol = results.save(epoch, metrics_symbol, 'split GER')
                save_position = results.save(epoch, metrics_position, 'split HER')

                if save_symbol:
                    model_path = '{}/{}_model_symbol'.format(logger.folder, experiment)
                    print('Saving symbol model to {}'.format(model_path))
                    saver_symbol.save(sess, model_path, global_step=epoch, latest_filename='checkpoint_symbol')

                if save_position:
                    model_path = '{}/{}_model_position'.format(logger.folder, experiment)
                    print('Saving position model to {}'.format(model_path))
                    saver_position.save(sess, model_path, global_step=epoch, latest_filename='checkpoint_position')

        # ===============================================
        # Load best split models
        model_path = tf.train.latest_checkpoint(logger.folder, 'checkpoint_symbol')
        saver_symbol.restore(sess, model_path)
        print('Restored best symbol model')

        model_path = tf.train.latest_checkpoint(logger.folder, 'checkpoint_position')
        saver_position.restore(sess, model_path)
        print('Restored best position model')

    # ===============================================
    # Joint training
    for epoch in range(1, FLAGS.joint_epochs+1):
        print("Joint epoch {}/{}".format(epoch, FLAGS.joint_epochs))

        it_train = train_ds.make_one_shot_iterator()
        next_batch = it_train.get_next()
        batch = 1
        while True:
            try:
                X, XL, Y_symbol, Y_position, Y_joint, YL = prepare_data(
                    sess.run(next_batch),
                    vocabulary_sizes,
                    params
                )

                print('Batch {}: {} samples'.format(batch, len(X)))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops): # Enables batch normalization
                    _ = sess.run(
                        optimizer_joint,
                        {
                            model_placeholders['input']: X,
                            model_placeholders['seq_len']: XL,
                            model_placeholders['joint']['target']: sparse_tuple_from(Y_joint),
                            model_placeholders['keep_prob']: 0.75
                        }
                    )

                batch = batch + 1
            except tf.errors.OutOfRangeError:
                break
        
        # ===============================================
        # Split validation   
        if epoch % 5 == 0:
            metrics_joint = (0, 0, 0) # (editions, total_length, sequences)

            it_val = val_ds.make_one_shot_iterator()
            next_batch = it_val.get_next()
            while True:
                try:
                    X, XL, Y_symbol, Y_position, Y_joint, YL = prepare_data(
                        sess.run(next_batch),
                        vocabulary_sizes,
                        params
                    )

                    pred_joint = sess.run(
                        decoder_joint,
                        {
                            model_placeholders['input']: X,
                            model_placeholders['seq_len']: XL,
                            model_placeholders['keep_prob']: 1.0,
                        }
                    )

                    metrics_joint, H, Y = eval(pred_joint, Y_joint, vocabularies[2], metrics_joint)
                    logger.log_predictions(epoch, H, Y)

                except tf.errors.OutOfRangeError:
                    break

            save_joint = results.save(epoch, metrics_joint, 'joint SER')

            if save_joint:
                model_path = '{}/{}_model_joint'.format(logger.folder, experiment)
                print('Saving joint model to {}'.format(model_path))
                saver_joint.save(sess, model_path, global_step=epoch, latest_filename='checkpoint_joint')
