import argparse
import tensorflow as tf
import json
from sklearn.utils import shuffle
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import numpy as np
import cv2
import os
import sys

# ===================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.reset_default_graph()
sess = tf.InteractiveSession(config=config)

# ===================================================


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
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


def build_batch(v,channels):
    batch_size = len(v)

    # for CTC
    image_widths = [img.shape[1] for img in v]
    image_height = [img.shape[0] for img in v]

    max_image_height = max(image_height)
    max_image_width = max(image_widths)

    batch = np.zeros(shape=[batch_size, max_image_height, max_image_width, channels], dtype=np.float32)

    for i, img in enumerate(v):
        if channels == 1:
            batch[i, 0:img.shape[0], 0:img.shape[1],0] = img
        else:
            batch[i, 0:img.shape[0],0:img.shape[1]] = img

    return batch


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


def load_set(data_filepath):
    lines = open(data_filepath, 'r').read().splitlines()

    vocabulary = set()    
    x = []
    y = []

    for line in lines:
        page_path, json_path = line.split('\t')
        page_image = cv2.imread(page_path,True)
        
        print('Loading', json_path)
        if page_image is not None:
            with open(json_path) as json_file:  
                data = json.load(json_file)
                
                for page in data['pages']:
                    if "regions" in page:
                        for region in page['regions']:
                            if region['type'] == 'staff' and "symbols" in region:                            
                                symbol_sequence = [s["agnostic_symbol_type"] + ":" + s["position_in_straff"] for s in region["symbols"] ]
                                # TODO: mover dentro del if
                                y.append(symbol_sequence)
                                vocabulary.update(symbol_sequence)                            
                                top, left, bottom, right = region["bounding_box"]["fromY"], region["bounding_box"]["fromX"], region["bounding_box"]["toY"], region["bounding_box"]["toX"]
                            
                                region_image = page_image[top:bottom,left:right]
                                if region_image is not None:
                                    x.append(region_image)

    w2i = {}
    i2w = {}
    
    for idx, symbol in enumerate(vocabulary):
        w2i[symbol] = idx
        i2w[idx] = symbol     
        
        
    # Data normalization 
    for i in range(min(len(x),len(y))):
        for idx, symbol in enumerate(y[i]):
            y[i][idx] = w2i[symbol]
        
    return x, y, w2i, i2w


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
    

def data_preparation(X, Y, params):
    height = params['img_height']

    for i in range(min(len(X),len(Y))):
        img = cv2.cvtColor(X[i], cv2.COLOR_BGR2GRAY) # Pre-process
        img = (255. - img) / 255.
        width = int(float(height * img.shape[1]) / img.shape[0])

        X[i] = cv2.resize(img, (width, height))

    return X, Y

def data_augmentation(X,Y):
    augmented_X = []
    augmented_Y = []
  
    
    for idx, x in enumerate(X):        
        for i in [1, 7]:
            x_i = cv2.fastNlMeansDenoising(x,None,i)            
            augmented_X.append(x_i)
            augmented_Y.append(Y[idx])        
            
    return augmented_X, augmented_Y
        

# ==============================================================
#
#                           MAIN
#
# ==============================================================

if __name__ == "__main__":

    # ========
    max_epochs = 1000
    mini_batch_size = 16
    val_split = 0.1
    fixed_height = 64
    # ========

    parser = argparse.ArgumentParser(description='CRNN Training for HMR.')
    parser.add_argument('-data', dest='data', type=str, required=True, help='Path to data file.')
    parser.add_argument('-save_model', dest='save_model', type=str, default=None, help='Path to saved model.')
    parser.add_argument('-vocabulary', dest='vocabulary', type=str, required=True, help='Path to export the vocabulary.')
    args = parser.parse_args()


    # ===============================================
    # Loading data
    print("Loading data...")

    X, Y, w2i, i2w  = load_set(args.data)

    vocabulary_size = len(w2i)
    save_vocabulary(args.vocabulary, w2i)
        
    X, Y = shuffle(X, Y)

    val_idx = int(val_split * len(X))
    X_train, Y_train = X[val_idx:], Y[val_idx:]
    X_val, Y_val = X[:val_idx], Y[:val_idx]

    print("Done")


    # ===============================================
    # CRNN
    print("Creating model...")
    
    params = default_model_params(fixed_height, vocabulary_size)    
   
    crnn_placeholders = crnn(params)
    optimizer = tf.train.AdamOptimizer().minimize(crnn_placeholders['loss'])
    decoder, log_prob = tf.nn.ctc_greedy_decoder(crnn_placeholders['logits'], crnn_placeholders['seq_len'])

    print("Done")
    
    # ===============================================
    # Data preparation
    print("Preparing data...")
    
    X_train, Y_train = data_augmentation(X_train,Y_train)
    X_train, Y_train = shuffle(X_train, Y_train)
    
    X_train, Y_train = data_preparation(X_train, Y_train, params)        # TODO: replace with data augmentation WIP
    L_train = [image.shape[1] // params['width_reduction'] for image in X_train]
    X_train = build_batch(X_train, channels = 1)
    
    X_val, Y_val = data_preparation(X_val, Y_val, params)
    L_val = [image.shape[1] // params['width_reduction'] for image in X_val]
    X_val = build_batch(X_val, channels = 1)

    print("Done")
    
    # ===============================================
    # Training   

    print('Training with ' + str(X_train.shape[0]) + ' samples.')
    print('Validating with ' + str(X_val.shape[0]) + ' samples.')

    saver = tf.train.Saver(max_to_keep=None)
    sess.run(tf.global_variables_initializer())

    for epoch in range(max_epochs):
        print("Epoch {}/{}".format(epoch, max_epochs))
        epoch_loss = 0

        for batch_idx in range(0, X_train.shape[0], mini_batch_size):
            X_train_batch = X_train[batch_idx:batch_idx + mini_batch_size]
            L_train_batch = L_train[batch_idx:batch_idx + mini_batch_size]
            Y_train_batch = Y_train[batch_idx:batch_idx + mini_batch_size]


            # Deal with empty staff sections
            for idx, _ in enumerate(X_train_batch):
                if len(Y_train_batch[idx]) == 0:
                    Y_train_batch[idx] = [vocabulary_size]  # Blank CTC

            _ = sess.run(optimizer, 
                         {
                             crnn_placeholders['input']: X_train_batch,
                             crnn_placeholders['seq_len']: L_train_batch,
                             crnn_placeholders['target']: sparse_tuple_from(Y_train_batch),
                             crnn_placeholders['keep_prob']: 0.5
                         }
                        )


        # Validation
        if epoch % 5 == 0:
            acc_ed = 0
            acc_count = 0
            acc_len = 0

            for batch_idx in range(0, X_val.shape[0], mini_batch_size):

                pred = sess.run(decoder,
                                {
                                    crnn_placeholders['input']: X_val[batch_idx:batch_idx + mini_batch_size],
                                    crnn_placeholders['seq_len']: L_val[batch_idx:batch_idx + mini_batch_size],
                                    crnn_placeholders['keep_prob']: 1.0,
                                }
                               )

                sequence = sparse_tensor_to_strs(pred)
                for i in range(len(sequence)):                    
                    h = [ i2w[w] for w in sequence[i] ]
                    y = [ i2w[w] for w in Y_val[batch_idx+i] ]

                    print("Y:",y) # ************
                    print("H:",h) # ************
                    
                    acc_ed += edit_distance(h, y)
                    acc_len += len(y)
                    acc_count += 1

            print('Epoch',epoch,' - SER:', str(100. * acc_ed / acc_len), ' - From ',acc_count,'samples')
            
            if epoch % 5 == 0:
                if args.save_model is not None:
                    save_model_epoch = args.save_model+'_'+str(epoch)
                    print('-> Saving current model to ',save_model_epoch)
                    saver.save(sess, save_model_epoch)

