import argparse
import cv2
from datetime import datetime
import json
import logging
from math import ceil
import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from threading import Lock


# ===================================================

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for h in log.handlers:
    h.setFormatter(formatter)

# create file handler which logs even debug messages
#fh = logging.FileHandler('tensorflow.log')
#fh.setLevel(logging.DEBUG)
#fh.setFormatter(formatter)
#log.addHandler(fh)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# ===================================================


class ImageCache:
    def __init__(self):
        self.__lock = Lock()
        self.__images = {}

    def read_image(self, path):
        self.__lock.acquire()
        image = []
        if path in self.__images:
            logging.debug('-- Reading {} from cache'.format(path))
            image = self.__images[path]
        else:
            logging.debug('-- Reading {} from disk'.format(path))
            image = cv2.imread(path, True)
            self.__images[path] = image
        self.__lock.release()
        return image


# ===================================================


# https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb
# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa 
# (e.g., 5 -> "dad") for each language,
class LanguageIndex():
    def __init__(self, lang):
        self.__lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()

        self.create_index()
    
    def create_index(self):
        for sequence in self.__lang:
            self.vocab.update(sequence)
            
        self.vocab = sorted(self.vocab)
    
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index
    
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
            

# ===================================================


class LstReader:
    def __init__(self, lst_path):
        self.images = []
        self.regions = []
        self.symbols = []
        self.positions = []
        self.joint = []
        self.__load_lst(lst_path)

    def __load_lst(self, lst_path):
        lines = open(lst_path, 'r').read().splitlines()
        for line in lines:
            page_path, json_path = line.split('\t')
            with open(json_path) as json_file:
                region_count = 0
                data = json.load(json_file)               
                for page in data['pages']:
                    if 'regions' in page:
                        for region in page['regions']:
                            if region['type'] == 'staff' and 'symbols' in region:
                                self.images.append(page_path)
                                self.symbols.append(['<s>'] + [s['agnostic_symbol_type'] for s in region['symbols']] + ['<e>'])
                                self.positions.append(['<s>'] + [s["position_in_straff"] for s in region['symbols']] + ['<e>'])
                                self.joint.append(['<s>'] + ['{}:{}'.format(s['agnostic_symbol_type'], s["position_in_straff"]) for s in region['symbols']] + ['<e>'])
                                top, left, bottom, right = region['bounding_box']['fromY'], region['bounding_box']['fromX'], region['bounding_box']['toY'], region['bounding_box']['toX']
                                self.regions.append([top, bottom, left, right])
                                region_count += 1
                logging.info('{}: {} regions'.format(json_path, region_count))
                if region_count == 0:
                    logging.warning('No regions found in {}'.format(json_path))
        
        self.symbol_lang = LanguageIndex(self.symbols)
        for i, seq in enumerate(self.symbols):
            self.symbols[i] = [self.symbol_lang.word2idx[word] for word in seq]
        
        self.position_lang = LanguageIndex(self.positions)
        for i, seq in enumerate(self.positions):
            self.positions[i] = [self.position_lang.word2idx[word] for word in seq]

        self.joint_lang = LanguageIndex(self.joint)
        for i, seq in enumerate(self.joint):
            self.joint[i] = [self.joint_lang.word2idx[word] for word in seq]
    

# ===================================================


class DataReader:
    def __init__(self, lst_path, image_height, channels=1, test_split=0.1, validation_split=0.1, batch_size=16, parallel=tf.data.experimental.AUTOTUNE):
        self.__lst = LstReader(lst_path)
        self.__cache = ImageCache()
        self.__IMAGE_HEIGHT = image_height
        self.__CHANNELS = channels
        self.__PARALLEL = parallel
        self.__BATCH_SIZE = batch_size
        self.DATA_SIZE = len(self.__lst.regions)
        self.TEST_SPLIT = np.uint32(self.DATA_SIZE * test_split)
        self.VAL_SPLIT = np.uint32((self.DATA_SIZE - self.TEST_SPLIT) * validation_split)
        self.TRAIN_SPLIT = np.uint32(self.DATA_SIZE - self.TEST_SPLIT - self.VAL_SPLIT)
        
        images, regions, symbols, positions, joint = shuffle(self.__lst.images,
                                                            self.__lst.regions,
                                                            self.__lst.symbols,
                                                            self.__lst.positions,
                                                            self.__lst.joint)

        self.__image_ds = tf.data.Dataset.from_tensor_slices(images)
        self.__region_ds = tf.data.Dataset.from_generator(lambda: regions, tf.int32) # Workaround for creating a dataset with sequences of different length
        self.__symbol_ds = tf.data.Dataset.from_generator(lambda: symbols, tf.int32)
        self.__position_ds = tf.data.Dataset.from_generator(lambda: positions, tf.int32)
        self.__joint_ds = tf.data.Dataset.from_generator(lambda: joint, tf.int32)

    def __make_partitions(self, dataset):
        shapes = (tf.TensorShape([self.__IMAGE_HEIGHT, None, 1]), tf.TensorShape(()), tf.TensorShape([None]), tf.TensorShape(()))

        test_ds = dataset \
            .take(self.TEST_SPLIT) \
            .map(self.__map_load_and_preprocess_regions, num_parallel_calls=self.__PARALLEL) \
            .shuffle(buffer_size=self.TEST_SPLIT) \
            .padded_batch(self.__BATCH_SIZE, padded_shapes=shapes, padding_values=(0., 0, -1, 0)) \
            .prefetch(1)

        val_ds = dataset \
            .skip(self.TEST_SPLIT) \
            .take(self.VAL_SPLIT) \
            .map(self.__map_load_and_preprocess_regions, num_parallel_calls=self.__PARALLEL) \
            .shuffle(buffer_size=self.VAL_SPLIT) \
            .padded_batch(self.__BATCH_SIZE, padded_shapes=shapes, padding_values=(0., 0, -1, 0)) \
            .prefetch(1)

        train_ds = dataset \
            .skip(self.TEST_SPLIT + self.VAL_SPLIT) \
            .map(self.__map_load_and_preprocess_regions, num_parallel_calls=self.__PARALLEL) \
            .shuffle(buffer_size=self.TRAIN_SPLIT) \
            .padded_batch(self.__BATCH_SIZE, padded_shapes=shapes, padding_values=(0., 0, -1, 0)) \
            .prefetch(1)
        
        return train_ds, val_ds, test_ds
        
    def get_symbol_data(self):
        dataset = tf.data.Dataset.zip((self.__image_ds, self.__region_ds, self.__symbol_ds))        
        train_ds, val_ds, test_ds = self.__make_partitions(dataset)
        return train_ds, val_ds, test_ds, self.__lst.symbol_lang
    
    def get_position_data(self):
        dataset = tf.data.Dataset.zip((self.__image_ds, self.__region_ds, self.__position_ds))
        train_ds, val_ds, test_ds = self.__make_partitions(dataset)
        return train_ds, val_ds, test_ds, self.__lst.position_lang

    def get_joint_data(self):
        dataset = tf.data.Dataset.zip((self.__image_ds, self.__region_ds, self.__joint_ds))        
        train_ds, val_ds, test_ds = self.__make_partitions(dataset)
        return train_ds, val_ds, test_ds, self.__lst.joint_lang

    def __load_and_preprocess_regions(self, path, region, label):
        logging.debug('Loading region {}[{}]'.format(path, region))
        page_image = self.__cache.read_image(path.decode())
        top, bottom, left, right = region

        img = page_image[top:bottom, left:right]
        if self.__CHANNELS == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Pre-process
        img = np.float32((255. - img) / 255.)
        height = self.__IMAGE_HEIGHT
        width = int(float(height * img.shape[1]) / img.shape[0])
        img = cv2.resize(img, (width, height))
        if self.__CHANNELS == 1:
            img = img[:, :, np.newaxis]

        return img, np.int32(width), label, np.int32(len(label))
    
    def __map_load_and_preprocess_regions(self, image, region, label):
        return tf.py_func(self.__load_and_preprocess_regions, [image, region, label], [tf.float32, tf.int32, tf.int32, tf.int32])


# ===================================================


def leaky_relu(features, alpha=0.2, name=None):
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features)
    alpha = ops.convert_to_tensor(alpha)
    return math_ops.maximum(alpha * features, features)


def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = 16
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [64, 64, 128, 128]
    params['conv_filter_size'] = [[5, 5], [5, 5], [3, 3], [3, 3]]
    params['conv_pooling_size'] = [[2, 2], [2, 1], [2, 1], [2, 1]]
    params['rnn_units'] = 256
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size

    width_reduction = 1
    for i in range(params['conv_blocks']):
        width_reduction = width_reduction * params['conv_pooling_size'][i][1]

    params['width_reduction'] = width_reduction

    return params


def crnn(params):
    input = tf.placeholder(shape=(None,
                                  params['img_height'],
                                  params['img_width'],
                                  params['img_channels']),  # [batch, height, width, channels]
                           dtype=tf.float32,
                           name='model_input')

    input_shape = tf.shape(input)

    width_reduction = 1
    height_reduction = 1

    # Convolutional blocks
    x = input
    for i in range(params['conv_blocks']):
        x = tf.layers.conv2d(
            inputs=x,
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same",
            activation=None)

        x = tf.layers.batch_normalization(x)

        x = leaky_relu(x)

        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=params['conv_pooling_size'][i],
                                    strides=params['conv_pooling_size'][i],
                                    name='conv_'+str(i))

        width_reduction = width_reduction * params['conv_pooling_size'][i][1]
        height_reduction = height_reduction * params['conv_pooling_size'][i][0]

    # Prepare output of conv block for recurrent blocks
    features = tf.transpose(x, perm=[2, 0, 3, 1])  # -> [width, batch, height, channels] (time_major=True)
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
    feature_width = input_shape[2] // width_reduction
    features = tf.reshape(features, tf.stack([tf.cast(feature_width, 'int32'), input_shape[0],
                                              tf.cast(feature_dim, 'int32')]))  # -> [width, batch, features]

    tf.constant(params['img_height'], name='input_height')
    tf.constant(width_reduction, name='width_reduction')

    # Recurrent block
    rnn_keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(rnn_hidden_units),
                                           input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(rnn_hidden_units),
                                           input_keep_prob=rnn_keep_prob)
             for _ in range(rnn_hidden_layers)]),
        features,
        dtype=tf.float32,
        time_major=True,
    )

    rnn_outputs = tf.concat(rnn_outputs, 2)

    logits = tf.layers.dense(rnn_outputs, params['vocabulary_size']+1) # +1 because of 'blank' CTC

    # Add softmax!
    softmax = tf.nn.softmax(logits)
    
    tf.add_to_collection("softmax", softmax) # for restoring purposes
    tf.add_to_collection("logits", logits)  # for restoring purposes

    # CTC Loss computation
    seq_len = tf.placeholder(tf.int32, [None], name='seq_lengths') # Real length of the image
    targets = tf.sparse_placeholder(dtype=tf.int32, name='target')
    ctc_loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len, time_major=True)
    loss = tf.reduce_mean(ctc_loss)


    return {'input': input,
            'seq_len': seq_len,
            'target': targets,
            'logits': logits,
            'softmax':softmax,
            'loss': loss,
            'keep_prob': rnn_keep_prob}


def save_vocabulary(filepath, vocabulary):
    np.save(filepath,vocabulary)
      

# ===================================================


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representation of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def sparse_tensor_to_strs(sparse_tensor):
    indices= sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [ [] for i in range(dense_shape[0]) ]

    string = []
    ptr = 0
    b = 0

    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]

        string.append(values[ptr])

        ptr = ptr + 1

    strs[b] = string

    return strs



def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def edit_distance(a,b,EOS=-1,PAD=-1):
    _a = [s for s in a if s != EOS and s != PAD]
    _b = [s for s in b if s != EOS and s != PAD]

    return levenshtein(_a,_b)


# ===================================================


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CRNN Training for HMR.')
    
    parser.add_argument('--input-data', dest='data_path', type=str, required=True, help='Path to data file.')
    parser.add_argument('--output-vocabulary', dest='vocabulary_path', required=True, help='Saves vocabulary file to the specified path')
    parser.add_argument('--save-model', dest='save_model', type=str, default=None, help='Path to saved model.')

    parser.add_argument('--image-height', dest='image_height', type=int, default=64, help='Image size will be reduced to this height')
    parser.add_argument('--channels', dest='channels', type=int, default=1, help='Number of channels in training')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1000, help='Number of training epochs')

    FLAGS = parser.parse_args()


    # ===============================================
    # Loading data
    log.info('Preparing data...')
    
    data_reader = DataReader(FLAGS.data_path, image_height=FLAGS.image_height, channels=FLAGS.channels)
    train_ds, val_ds, test_ds, lang = data_reader.get_joint_data()

    vocabulary_size = len(lang.word2idx)

    log.info('Done')

    # ===============================================
    # Setting params
    params = default_model_params(FLAGS.image_height, vocabulary_size)    

    # ===============================================
    # CRNN
    log.info("Creating model...")
       
    crnn_placeholders = crnn(params)
    optimizer = tf.train.AdamOptimizer().minimize(crnn_placeholders['loss'])
    decoder, log_prob = tf.nn.ctc_greedy_decoder(crnn_placeholders['logits'], crnn_placeholders['seq_len'])

    log.info("Done")

    # ===============================================
    # Training   
    log.info('Training with ' + str(data_reader.TRAIN_SPLIT) + ' samples.')
    log.info('Validating with ' + str(data_reader.VAL_SPLIT) + ' samples.')
    log.info('Testing with ' + str(data_reader.TEST_SPLIT) + ' samples.')

    saver = tf.train.Saver(max_to_keep=None)
    sess.run(tf.global_variables_initializer())

    for epoch in range(FLAGS.epochs):
        log.info("Epoch {}/{}".format(epoch, FLAGS.epochs))

        it_train = train_ds.make_one_shot_iterator()
        next_batch = it_train.get_next()
        batch = 1
        while True:
            try:
                X_train_batch, XL_train_batch, Y_train_batch, YL_train_batch = sess.run(next_batch)
                XL_train_batch = [length // params['width_reduction'] for length in XL_train_batch]
                Y_train_batch = [y[:YL_train_batch[idx]] for idx, y in enumerate(Y_train_batch)]

                log.info('Batch {}/{}: {} samples'.format(batch, ceil(data_reader.TRAIN_SPLIT/FLAGS.batch_size), len(X_train_batch)))

                # Deal with empty staff sections
                for idx, _ in enumerate(X_train_batch):
                    if YL_train_batch[idx] == 0:
                        Y_train_batch[idx] = [vocabulary_size]  # Blank CTC
                        YL_train_batch[idx] = 1

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    _ = sess.run(optimizer, 
                                {
                                    crnn_placeholders['input']: X_train_batch,
                                    crnn_placeholders['seq_len']: XL_train_batch,
                                    crnn_placeholders['target']: sparse_tuple_from(Y_train_batch),
                                    crnn_placeholders['keep_prob']: 0.75
                                }
                                )

                batch = batch + 1
            except tf.errors.OutOfRangeError:
                break
        
        # Validation
        if epoch % 5 == 0:
            acc_ed = 0
            acc_count = 0
            acc_len = 0

            it_val = val_ds.make_one_shot_iterator()
            next_batch = it_val.get_next()
            while True:
                try:
                    X_val_batch, XL_val_batch, Y_val_batch, YL_val_batch = sess.run(next_batch)
                    XL_val_batch = [length // params['width_reduction'] for length in XL_val_batch]
                    Y_val_batch = [y[:YL_val_batch[idx]] for idx, y in enumerate(Y_val_batch)]

                    pred = sess.run(decoder,
                                    {
                                        crnn_placeholders['input']: X_val_batch,
                                        crnn_placeholders['seq_len']: XL_val_batch,
                                        crnn_placeholders['keep_prob']: 1.0,
                                    }
                                )

                    sequence = sparse_tensor_to_strs(pred)
                    for i in range(len(sequence)):                    
                        h = [ lang.idx2word[w] for w in sequence[i] ]
                        y = [ lang.idx2word[w] for w in Y_val_batch[i] ]

                        log.info("Y:{}".format(y)) # ************
                        log.info("H:{}".format(h)) # ************
                        
                        acc_ed += edit_distance(h, y)
                        acc_len += len(y)
                        acc_count += 1
                except tf.errors.OutOfRangeError:
                    break

            log.info('Epoch {} - SER: {} - From {} samples'.format(epoch, str(100. * acc_ed / acc_len), acc_count))
            
            if epoch % 5 == 0:
                if FLAGS.save_model is not None:
                    save_model_epoch = args.save_model+'_'+str(epoch)
                    log.info('-> Saving current model to {}'.format(save_model_epoch))
                    saver.save(sess, save_model_epoch)
