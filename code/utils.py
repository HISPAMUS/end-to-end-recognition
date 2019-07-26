from datetime import datetime
import logging
from model import sparse_tensor_to_strs
import numpy as np
import os
import tensorflow as tf


# ===================================================


def config(FLAGS):
    if FLAGS.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    sess = tf.InteractiveSession(config=config)
    return sess


# ===================================================


def save_vocabulary(filepath, vocabulary):
    np.save(filepath, vocabulary)


# ===================================================


# ===================================================


class Logger:
    def __init__(self, folder):
        self.folder = folder

        self.metrics = logging.getLogger('metrics')
        m_handler = logging.FileHandler(folder+'/metrics.log')
        m_handler.setLevel(logging.INFO)
        self.metrics.addHandler(m_handler)

        self.predictions = logging.getLogger('predictions')
        p_handler = logging.FileHandler(folder+'/predictions.log')
        p_handler.setLevel(logging.INFO)
        self.predictions.addHandler(p_handler)

        self.output = logging.getLogger('output')
        o_handler = logging.FileHandler(folder+'/output.log')
        o_handler.setLevel(logging.INFO)
        self.output.addHandler(o_handler)
    
    def log_metrics(self, epoch, metrics, metrics_name):
        log = self.metrics
        error_rate, samples = 100. * metrics[0] / metrics[1], metrics[2]
        msg = 'Epoch {} - {}: {:.2f} - From {} samples'.format(epoch, metrics_name, error_rate, samples)
        print(msg)
        log.error(msg)
    
    def log_predictions(self, epoch, H, Y):
        log = self.predictions
        log.error('Epoch {}'.format(epoch))
        for i in range(len(H)):
            log.error('H: {}'.format(H[i]))
            log.error('Y: {}'.format(Y[i]))
    
    def log(self, msg):
        log = self.output
        log.error(msg)


def get_logger(prefix, FLAGS):
    now = datetime.now()
    timestamp = now.strftime('%Y%m%d%H%M%S')

    # params = 'data_{}_seed_{}_freeze_{}_height_{}_channels_{}_augmentation_{}_delimiter_{}_test_{}_batch_{}'.format(
    #     os.path.splitext(os.path.basename(FLAGS.data_path))[0],
    #     FLAGS.seed,
    #     FLAGS.freeze,
    #     FLAGS.image_height,
    #     FLAGS.channels,
    #     FLAGS.image_transformations,
    #     FLAGS.sequence_delimiter,
    #     FLAGS.test_split,
    #     FLAGS.batch_size
    # )

    # if FLAGS.log is None:
    #     folder = 'logs/{}_pid_{}_{}_{}'.format(timestamp, os.getpid(), prefix, params)
    # else:
    folder = 'logs/{}_pid_{}_{}'.format(timestamp, os.getpid(), FLAGS.log)
    os.makedirs(folder)

    return Logger(folder)


class ResultsManager:
    def __init__(self, logger):
        self.logger = logger
        self.results = dict()
        self.best = dict()
    
    def save(self, epoch, metrics, metric_name):
        improved = True
        self.logger.log_metrics(epoch, metrics, metric_name)
        error_rate = float(metrics[0]) / float(metrics[1])
        if metric_name in self.results:
            self.results[metric_name].append((epoch, error_rate))
        else:
            self.results[metric_name] =  [(epoch, error_rate)]
        if metric_name in self.best:
            if error_rate < self.best[metric_name][1]:
                self.best[metric_name] = (epoch, error_rate)
            else:
                improved = False
        else:
            self.best[metric_name] = (epoch, error_rate)
        return improved


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


# ===================================================


def edit_distance(a,b,EOS=-1,PAD=-1):
    _a = [s for s in a if s != EOS and s != PAD]
    _b = [s for s in b if s != EOS and s != PAD]

    return levenshtein(_a,_b)


# ===================================================


def eval(predictions, labels, vocabulary, metrics):
    predictions = sparse_tensor_to_strs(predictions)
    H, Y = [], []
    for i in range(len(predictions)):                    
        h = [ vocabulary.idx2word[w] for w in predictions[i] ]
        y = [ vocabulary.idx2word[w] for w in labels[i] ]
        
        metrics = (
            metrics[0] + edit_distance(h, y), # number of edition operations
            metrics[1] + len(y), # total length of the sequences
            metrics[2] + 1 # number of sequences
        )

        H.append(h)
        Y.append(y)

    return metrics, H, Y
