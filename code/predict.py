import argparse
import tensorflow as tf
import cv2
import numpy as np

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

parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
parser.add_argument('-image',  dest='image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-model', dest='model', type=str, default='Model/Saved/hispamus_model_175.meta', help='Path to the trained model.')
parser.add_argument('-vocabulary', dest='vocabulary', type=str, default='Model/Saved/vocabulary.npy', help='Path to the vocabulary file.')
args = parser.parse_args()

import os

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.reset_default_graph()
sess = tf.InteractiveSession(config=config)

# Read the dictionary
word2int = np.load(args.vocabulary).item()     # Category -> int
int2word = dict((v, k) for k, v in word2int.items())     # int -> Category

#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph(args.model)
saver.restore(sess,args.model[:-5])

graph = tf.get_default_graph()
input = graph.get_tensor_by_name('model_input:0')
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

#decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
decoded = tf.nn.softmax(logits)

original_image = cv2.imread(args.image,True)

image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) # Pre-process
new_width = int(float(HEIGHT * image.shape[1]) / image.shape[0])
image = cv2.resize(image, (new_width, HEIGHT))

image = (255.-image)/255
image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]

prediction = sess.run(decoded,
                      feed_dict={
                          input: image,
                          seq_len: seq_lengths,
                          rnn_keep_prob: 1.0,
                      })

# prediction -> [frame, sample, character]

pred_per_frame = []
for i,v in enumerate(prediction):
        pred_per_frame.append(np.argmax(v[0]))

width_refactor = 1.
width_refactor *= original_image.shape[0]
width_refactor /= HEIGHT
width_refactor *= WIDTH_REDUCTION

# Process symbol and positions
result = []
prev = -1
for idx, w in enumerate(pred_per_frame):
    if w < len(int2word): # != BLANK
        if prev == -1:
            start = idx
        elif prev != w:
            if prev != len(int2word):
                result.append((int2word[prev],int(start*width_refactor),int(idx*width_refactor)))
            start = idx
    else: # == BLANK
        if prev != -1 and prev != len(int2word):
            result.append((int2word[prev],int(start*width_refactor),int(idx*width_refactor)))
    prev = w


for symbol, start, end in result:
    print(symbol, start, end)
