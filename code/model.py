import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


# ===================================================


def default_model_params(img_height, img_channels, vocabulary_sizes, batch_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = batch_size
    params['img_channels'] = img_channels
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [64, 64, 128, 128]
    params['conv_filter_size'] = [[5, 5], [5, 5], [3, 3], [3, 3]]
    params['conv_pooling_size'] = [[2, 2], [2, 1], [2, 1], [2, 1]]
    params['rnn_units'] = 256
    params['rnn_layers'] = 2
    params['vocabulary_sizes'] = vocabulary_sizes

    width_reduction = 1
    for i in range(params['conv_blocks']):
        width_reduction = width_reduction * params['conv_pooling_size'][i][1]

    params['width_reduction'] = width_reduction

    return params


# ===================================================


def leaky_relu(features, alpha=0.2, name=None):
    with ops.name_scope(name, "LeakyRelu", [features, alpha]):
        features = ops.convert_to_tensor(features)
        alpha = ops.convert_to_tensor(alpha)
        return math_ops.maximum(alpha * features, features)


# ===================================================


def cnn_block(x, params):
    # Convolutional blocks
    width_reduction = 1
    height_reduction = 1
    for i in range(params['conv_blocks']):
        x = tf.layers.conv2d(
            inputs=x,
            filters=params['conv_filter_n'][i],
            kernel_size=params['conv_filter_size'][i],
            padding="same",
            activation=None)

        x = tf.identity(x, name='conv_output_{}'.format(i))

        x = tf.layers.batch_normalization(x)

        x = leaky_relu(x)

        x = tf.layers.max_pooling2d(inputs=x,
                                    pool_size=params['conv_pooling_size'][i],
                                    strides=params['conv_pooling_size'][i],
                                    name='conv_'+str(i))

        width_reduction = width_reduction * params['conv_pooling_size'][i][1]
        height_reduction = height_reduction * params['conv_pooling_size'][i][0]

    return x, width_reduction, height_reduction


# ===================================================


def rnn_block(x, placeholders, params, vocabulary_index):
    # Recurrent block
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(rnn_hidden_units),
                                           input_keep_prob=placeholders['keep_prob'])
             for _ in range(rnn_hidden_layers)]),
        tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(rnn_hidden_units),
                                           input_keep_prob=placeholders['keep_prob'])
             for _ in range(rnn_hidden_layers)]),
        x,
        dtype=tf.float32,
        time_major=True,
    )

    rnn_outputs = tf.concat(rnn_outputs, 2)

    # +1 because of 'blank' CTC
    logits = tf.layers.dense(
        rnn_outputs, params['vocabulary_sizes'][vocabulary_index]+1)

    # Add softmax!
    softmax = tf.nn.softmax(logits)

    # tf.add_to_collection("softmax", softmax) # for restoring purposes
    # tf.add_to_collection("logits", logits)  # for restoring purposes

    # CTC Loss computation
    targets = tf.sparse_placeholder(dtype=tf.int32, name='target')
    ctc_loss = tf.nn.ctc_loss(labels=targets, inputs=logits,
                              sequence_length=placeholders['seq_len'], time_major=True)
    loss = tf.reduce_mean(ctc_loss)

    return {'target': targets,
            'logits': logits,
            'softmax': softmax,
            'loss': loss,
            'rnn_outputs': rnn_outputs}


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
    shape = np.asarray([len(sequences), np.asarray(
        indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


# ===================================================


def sparse_tensor_to_strs(sparse_tensor):
    indices = sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [[] for i in range(dense_shape[0])]

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
