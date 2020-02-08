# --------------------------------------------------------
# Tensorflow Implementation of Siamese Network
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------

import os
import tensorflow as tf

from solver import Solver

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_integer('batch_size', 256, 'batch size for one feed forwrad, default: 256')
tf.flags.DEFINE_float('lambda_1', 10., 'hyper-parameter for siamese loss for balancing total loss, default: 10')
tf.flags.DEFINE_string('dataset', 'fingervein', 'dataset name for choice [mnist|cifar10|fingervein], default: fingervein')

tf.flags.DEFINE_bool('is_train', True, 'training or inference mode, default: True')
tf.flags.DEFINE_bool('is_siamese', True, 'siamese network or inference mode, default: True')
tf.flags.DEFINE_float('learning_rate', 1e-4, 'initial learning rate, default: 0.0001')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for model to handle overfitting, default: 0.0001')
tf.flags.DEFINE_float('beta1', 0.5, 'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('margin', 5.0, 'margin of siamese network, default: 5.0')

tf.flags.DEFINE_integer('iters', 200000, 'number of iterations, default: 200000')
tf.flags.DEFINE_integer('print_freq', 1, 'print frequency for loss, default: 1')
tf.flags.DEFINE_integer('eval_freq', 10, 'evaluation frequency for test accuracy, default: 10')
tf.flags.DEFINE_integer('save_freq', 100, 'save frequency for model, default: 100')
tf.flags.DEFINE_integer('sample_freq', 100, 'sample frequency for saving image, default: 100')
tf.flags.DEFINE_integer('embedding_size', 512, 'number of sampling images for check generator quality, default: 512')
tf.flags.DEFINE_string('load_model', None, 'folder of saved model that you wish to test, (e.g. 20191214-1931), default: None')
tf.flags.DEFINE_float('threshold', None, 'threshold to test set for load_model, (e.g. 0.0718232), default: None')

FLAGS = tf.flags.FLAGS

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    solver = Solver(FLAGS)
    if FLAGS.is_train:
        solver.train()
    else:
        solver.test()

if __name__ == '__main__':
    tf.app.run()