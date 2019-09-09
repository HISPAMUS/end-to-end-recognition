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

    # Real length of the image
    seq_len = tf.placeholder(tf.int32, [None], name='seq_lengths')
    rnn_keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

    placeholders['input'] = input
    placeholders['seq_len'] = seq_len
    placeholders['keep_prob'] = rnn_keep_prob

    input_shape = tf.shape(input)

    with tf.variable_scope('joint'):
        with tf.variable_scope('cnn'):
            # Convolutional blocks
            x, width_reduction, height_reduction = cnn_block(input, params)

            # Prepare output of conv block for recurrent blocks
            # -> [width, batch, height, channels] (time_major=True)
            features = tf.transpose(x, perm=[2, 0, 3, 1])
            feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
            feature_width = input_shape[2] // width_reduction
            features = tf.reshape(features, tf.stack([tf.cast(feature_width, 'int32'), input_shape[0],
                                                      tf.cast(feature_dim, 'int32')]))  # -> [width, batch, features]

            # Used for prediction
            #tf.constant(params['img_height'], name='input_height')
            #tf.constant(width_reduction, name='width_reduction')

        with tf.variable_scope('rnn'):
            placeholders['joint'] = rnn_block(features, placeholders, params, 2)

    return placeholders


# ===================================================


if __name__ == '__main__':

    experiment = "exp01"

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
    parser.add_argument('--epochs', dest='epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--gpu', dest='gpu', type=str, default=None, help='GPU id')
    parser.add_argument('--seed', dest='seed', type=int, default=None, help='Random seed for shuffling data (default=None)')
    parser.add_argument('--freeze', dest='freeze', type=int, default=0, help='Unused')
    parser.add_argument('--train-limit', dest='train_limit', type=int, default=None, help='Number of training samples to use')

    # Misc options
    parser.add_argument('--log', dest='log', type=str, required=True, help='Log folder')

    FLAGS = parser.parse_args()

    # ===============================================
    # Initialize logger & results manager
    logger = get_logger(experiment, FLAGS)
    results = ResultsManager(logger)

    # ===============================================
    # Initialize TensorFlow
    sess = config(FLAGS)

    # ===============================================
    # Loading data
    logger.log('Preparing data...')

    data_reader = DataReader(
        FLAGS.data_path,
        image_height=FLAGS.image_height,
        channels=FLAGS.channels,
        sequence_delimiter=FLAGS.sequence_delimiter,
        test_split=FLAGS.test_split,
        batch_size=FLAGS.batch_size,
        image_transformations=FLAGS.image_transformations,
        seed=FLAGS.seed,
        train_limit=FLAGS.train_limit
    )

    train_ds, val_ds, test_ds = data_reader.get_data()
    # 0 -> symbol, 1 -> position, 2 -> joint
    vocabularies = data_reader.get_dictionaries()
    vocabulary_sizes = (
        len(vocabularies[0].word2idx),
        len(vocabularies[1].word2idx),
        len(vocabularies[2].word2idx)
    )

    logger.log('Done')

    # ===============================================
    # Setting params
    params = default_model_params(
        FLAGS.image_height, FLAGS.channels, vocabulary_sizes, FLAGS.batch_size)

    # ===============================================
    # CRNN
    logger.log("Creating model...")

    model_placeholders = model(params)

    optimizer_joint = tf.train.AdamOptimizer().minimize(model_placeholders['joint']['loss'])
    decoder_joint, log_prob_joint = tf.nn.ctc_greedy_decoder(
        model_placeholders['joint']['logits'],
        model_placeholders['seq_len']
    )

    logger.log("Done")

    # ===============================================
    # Training
    logger.log('Training with ' + str(data_reader.TRAIN_SPLIT) + ' samples.')
    logger.log('Validating with ' + str(data_reader.VAL_SPLIT) + ' samples.')
    logger.log('Testing with ' + str(data_reader.TEST_SPLIT) + ' samples.')

    saver = tf.train.Saver(max_to_keep=1)  # Saves the complete model

    sess.run(tf.global_variables_initializer())

    # ===============================================
    # Joint training
    for epoch in range(1, FLAGS.epochs+1):
        logger.log("Joint epoch {}/{}".format(epoch, FLAGS.epochs))

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

                logger.log('Batch {}: {} samples'.format(batch, len(X)))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # Enables batch normalization
                with tf.control_dependencies(update_ops):
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
        # Validation
        if epoch % 5 == 0:
            metrics_joint = (0, 0, 0)  # (editions, total_length, sequences)

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

            save_joint = results.save(epoch, metrics_joint, 'Validation SER')

            if save_joint:
                model_path = '{}/{}_model'.format(logger.folder, experiment)
                logger.log('Saving model to {}'.format(model_path))
                saver.save(sess, model_path, global_step=epoch)

    if FLAGS.test_split > 0:
        # ===============================================
        # Load best model
        model_path = tf.train.latest_checkpoint(logger.folder, 'checkpoint')
        saver.restore(sess, model_path)
        logger.log('Restored best model')

        # ===============================================
        # Validation
        metrics = (0, 0, 0)  # (editions, total_length, sequences)

        it = test_ds.make_one_shot_iterator()
        next_batch = it.get_next()
        while True:
            try:
                X, XL, Y_symbol, Y_position, Y_joint, YL = prepare_data(
                    sess.run(next_batch),
                    vocabulary_sizes,
                    params
                )

                pred = sess.run(
                    decoder_joint,
                    {
                        model_placeholders['input']: X,
                        model_placeholders['seq_len']: XL,
                        model_placeholders['keep_prob']: 1.0,
                    }
                )

                metrics, H, Y = eval(pred, Y_joint, vocabularies[2], metrics)
                logger.log_predictions(epoch, H, Y)

            except tf.errors.OutOfRangeError:
                break

        results.save(epoch, metrics, 'Test SER')
