# --------------------------------------------------------
# Tensorflow Implementation of Siamese Network
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------

import collections
import os
import sys
import time
import numpy as np
import logging
import math
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.manifold import TSNE

import tensorflow_utils as tf_utils
import utils as utils
# from dataset import Dataset

class Siamese(object):
    def __init__(self, sess, flags, image_size, Dataset):
        self.name = 'Siamese'
        self.sess = sess
        self.flags = flags
        self.image_size = image_size
        self.dataset = Dataset#Dataset(self.flags.dataset)
        self.is_train = self.flags.is_train
        self.is_siamese = self.flags.is_siamese

        self.learning_rate = self.flags.learning_rate
        self.weight_decay = self.flags.weight_decay
        self.iters = self.flags.iters
        self.start_decay_step = int(self.iters * 0.5)
        self.decay_steps = self.iters - self.start_decay_step
        self.beta1 = self.flags.beta1

        self.method = "ResNet18" #ResNet34
        self._ops = []
        self.keep_prob = 0.5

        self.margin = self.flags.margin
        self.sample_freq = self.flags.sample_freq
        self.embedding_size = self.flags.embedding_size

        # self.num_test = self.dataset.test_data.shape[0]#40
        self.conv_dims = self.set_conv_dims(self.method)
        self.accuracy = -1
        self.train_accuracy = -1############################

        self.use_batchnorm = True

        self.logger = logging.getLogger(__name__)  # logger
        self.logger.setLevel(logging.INFO)

        log_out_dir = "{}/log".format(self.flags.dataset)
        if not os.path.isdir(log_out_dir):
            os.makedirs(log_out_dir)

        self.logDir = log_out_dir

        utils.init_logger(logger=self.logger, logDir=self.logDir, isTrain=self.is_train, name=self.name)

        self._build_net(is_siamese=self.is_siamese)
        self._tensorboard()

        print('Initialized Siamese Network SUCCESS!')

    def _build_net(self, is_siamese=True):
        # build_graph
        self.train_mode = tf.compat.v1.placeholder(dtype=tf.dtypes.bool, name='train_mode_ph')

        if is_siamese:
            self.x1 = tf.placeholder(tf.float32, shape=[None, *self.image_size])
            self.x2 = tf.placeholder(tf.float32, shape=[None, *self.image_size])
            self.y_ = tf.placeholder(tf.float32, shape=[None])
            self.y_1 = tf.placeholder(tf.int32, shape=[None], name='label_1')
            self.y_2 = tf.placeholder(tf.int32, shape=[None], name='label_2')

            self.gw1, self.score1 = self.forward_network(self.x1, reuse=False)
            self.gw2, self.score2 = self.forward_network(self.x2, reuse=True)


        else:
            self.x1 = tf.placeholder(tf.float32, shape=[None, *self.image_size])
            self.y_1 = tf.placeholder(tf.int32, shape=[None], name='label_1')

            self.gw1, self.score1 = self.forward_network(self.x1, reuse=False)

        self.pred_cls = tf.math.argmax(self.score1, axis=1)
        self.loss = self._loss(is_siamese=is_siamese)

        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Optimizer
        train_op = self.optimizer_fn(self.loss, name='Adam')
        train_ops = [train_op] + self._ops
        self.optim = tf.group(*train_ops)
        # self.optim = tf.train.AdamOptimizer(learning_rate=self.flags.learning_rate, beta1=self.flags.beta1). \
        #     minimize(self.loss, var_list=vars)

    def optimizer_fn(self,loss, name=None):
        with tf.variable_scope(name):
            global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
            start_learning_rate = self.learning_rate
            end_learning_rate = 0.
            start_decay_step = self.start_decay_step
            decay_steps = self.decay_steps

            learning_rate = (tf.where(tf.greater_equal(global_step, start_decay_step),
                                      tf.train.polynomial_decay(start_learning_rate,
                                                                global_step - start_decay_step,
                                                                decay_steps, end_learning_rate, power=1.0),
                                      start_learning_rate))
            #self.tb_lr = tf.summary.scalar('learning_rate', learning_rate)

            learn_step = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1).minimize(loss, global_step=global_step)

        return learn_step

    def train_step(self, batch_imgs1, batch_label1, batch_imgs2, batch_label2, is_siamese=True):
        if is_siamese:
            batch_x1, batch_y1 = batch_imgs1, batch_label1
            batch_x2, batch_y2 = batch_imgs2, batch_label2
            batch_y = (batch_y1 == batch_y2).astype('float')

            feed = {self.x1: batch_x1,
                    self.x2: batch_x2,
                    self.y_: batch_y,
                    self.y_1: batch_y1,
                    self.y_2: batch_y2,
                    self.train_mode: True}

            _, total_loss, siamese_loss, reg_term, cls_loss_1, cls_loss_2, summary = self.sess.run(
                [self.optim, self.loss, self.siamese_loss, self.reg_term, self.cls_data_loss_1, self.cls_data_loss_2,
                 self.summary_op], feed_dict=feed)

        else:
            batch_x1, batch_y1 = batch_imgs1, batch_label1

            feed = {self.x1: batch_x1,
                    self.y_1: batch_y1,
                    self.train_mode: True}

            _, total_loss, reg_term, cls_loss_1, summary = self.sess.run(
                [self.optim, self.loss, self.reg_term, self.cls_data_loss_1, self.summary_op], feed_dict=feed)

            siamese_loss = None
            cls_loss_2 = None

        return total_loss, siamese_loss, reg_term, cls_loss_1, cls_loss_2, summary

    def forward_network(self, inputImg, padding='SAME', reuse=False):
        with tf.compat.v1.variable_scope(self.name, reuse=reuse):
            if self.method == 'mnist':
                # Stage 0
                tf_utils.print_activations(inputImg, logger=self.logger)
                s0_fc = tf_utils.flatten(x=inputImg, name='s0_flatten', logger=self.logger)
                s0_fc = tf_utils.linear(s0_fc, output_size=self.conv_dims[0],initializer='xavier',name='s0_fc',
                                        logger=self.logger)
                s0_fc = tf_utils.relu(s0_fc, name='s0_relu', logger=self.logger)

                # Stage 1
                # s1_fc = tf_utils.flatten(x=s0_fc, name='s1_flatten', logger=self.logger)
                s1_fc = tf_utils.linear(s0_fc, output_size=self.conv_dims[1],initializer='xavier',name='s1_fc',
                                        logger=self.logger)
                s1_fc = tf_utils.relu(s1_fc, name='s1_relu', logger=self.logger)

                # Stage 2
                # output = tf_utils.flatten(x=s1_fc, name='s2_flatten', logger=self.logger)
                output = tf_utils.linear(s1_fc, output_size=self.conv_dims[2],initializer='xavier',name='s2_fc',
                                        logger=self.logger)

            elif self.method == 'cifar10_v4':
                # Stage 0
                tf_utils.print_activations(inputImg, logger=self.logger)

                # Stage 1
                s1_conv1a = tf_utils.conv2d(x=inputImg, output_dim=self.conv_dims[0], k_h=1, k_w=1, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s1_conv1a', logger=self.logger)
                s1_conv1a = tf_utils.relu(s1_conv1a, name='s1_relu', logger=self.logger)

                # Stage 2
                s1_conv1 = tf_utils.conv2d(x=s1_conv1a, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s1_conv1', logger=self.logger)

                s1_conv1 = tf_utils.relu(s1_conv1, name='s2_relu', logger=self.logger)

                s1_conv1 = tf_utils.max_pool(x=s1_conv1, name='s1_maxpool', logger=self.logger)

                # Stage 3
                s2_conv1a = tf_utils.conv2d(x=s1_conv1, output_dim=self.conv_dims[2], k_h=1, k_w=1, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s2_conv1a', logger=self.logger)
                s2_conv1a = tf_utils.relu(s2_conv1a, name='s3_relu', logger=self.logger)

                # Stage 4
                s2_conv1 = tf_utils.conv2d(x=s2_conv1a, output_dim=self.conv_dims[3], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s2_conv1', logger=self.logger)
                s2_conv1 = tf_utils.relu(s2_conv1, name='s4_relu', logger=self.logger)

                # Stage 5
                s3_conv1a = tf_utils.conv2d(x=s2_conv1, output_dim=self.conv_dims[4], k_h=1, k_w=1, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s3_conv1a', logger=self.logger)
                s3_conv1a = tf_utils.relu(s3_conv1a, name='s5_relu', logger=self.logger)

                # Stage 6
                s3_conv1 = tf_utils.conv2d(x=s3_conv1a, output_dim=self.conv_dims[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s3_conv1', logger=self.logger)
                s3_conv1 = tf_utils.relu(s3_conv1, name='s6_relu', logger=self.logger)

                # Stage 7
                s4_conv1a = tf_utils.conv2d(x=s3_conv1, output_dim=self.conv_dims[6], k_h=1, k_w=1, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s4_conv1a', logger=self.logger)
                s4_conv1a = tf_utils.relu(s4_conv1a, name='s7_relu', logger=self.logger)

                # Stage 8
                s4_conv1 = tf_utils.conv2d(x=s4_conv1a, output_dim=self.conv_dims[7], k_h=3, k_w=3, d_h=1, d_w=1,
                                           padding=padding, initializer='He', name='s4_conv1', logger=self.logger)
                s4_conv1 = tf_utils.relu(s4_conv1, name='s8_relu', logger=self.logger)

                # Stage 9
                s5_conv1a = tf_utils.conv2d(x=s4_conv1, output_dim=self.conv_dims[8], k_h=1, k_w=1, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s5_conv1a', logger=self.logger)
                s5_conv1a = tf_utils.relu(s5_conv1a, name='s9_relu', logger=self.logger)
                s5_conv1a = tf_utils.max_pool(x=s5_conv1a, name='s5_maxpool', logger=self.logger)

                # Stage 10
                s5_flatten = tf_utils.flatten(x=s5_conv1a, name='s5_flatten', logger=self.logger)
                s5_flatten = tf_utils.linear(s5_flatten, output_size=self.conv_dims[9], initializer='xavier',
                                             name='s5_fc', logger=self.logger)
                s5_flatten = tf_utils.relu(s5_flatten, name='s10_relu', logger=self.logger)


                # Stage 11
                s6_flatten = tf_utils.linear(s5_flatten, output_size=self.conv_dims[10], initializer='xavier',
                                             name='s6_fc', logger=self.logger)
                s6_flatten = tf_utils.relu(s6_flatten, name='s11_relu', logger=self.logger)

                # Stage 12
                output = tf_utils.linear(s6_flatten, output_size=self.conv_dims[11], initializer='xavier', name='s7_fc',
                                         logger=self.logger)
                output_relu = tf_utils.relu(output, name='s12_relu', logger=self.logger)

                # Stage 14
                score = tf_utils.linear(output_relu, output_size=self.conv_dims[12], initializer='xavier', name='s8_fc',
                                        logger=self.logger)

            elif self.method == 'fingervein':
                # Stage 0
                tf_utils.print_activations(inputImg, logger=self.logger)

                # Stage 1
                input = tf_utils.conv2d(x=inputImg, output_dim=self.conv_dims[0], k_h=5, k_w=5, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s1_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s1_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s1_maxpool', logger=self.logger)

                # Stage 2
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[1], k_h=5, k_w=5, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s2_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s2_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s2_maxpool', logger=self.logger)

                # Stage 3
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[2], k_h=5, k_w=5, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s3_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s3_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s3_maxpool', logger=self.logger)

                # Stage 4
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[3], k_h=5, k_w=5, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s4_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s4_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s4_maxpool', logger=self.logger)

                # Stage 5
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[4], k_h=5, k_w=5, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s5_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s5_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s5_maxpool', logger=self.logger)

                # Stage 6
                _, h, w, _ = input.get_shape().as_list()
                input = tf_utils.avg_pool(input, name='gap', ksize=[1, h, w, 1], strides=[1, 1, 1, 1],
                                           logger=self.logger)
                output = tf_utils.flatten(x=input, name='s6_flatten', logger=self.logger)
                output_relu = tf_utils.relu(output, name='s6_relu', logger=self.logger)

                # Stage 7
                score = tf_utils.linear(output_relu, output_size=self.dataset.num_class, initializer='xavier',
                                        name='s7_fc', logger=self.logger)

            elif self.method == 'fingervein_v2':
                # Stage 0
                tf_utils.print_activations(inputImg, logger=self.logger)

                # Stage 1
                input = tf_utils.conv2d(x=inputImg, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s1_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s1_relu', logger=self.logger)

                # Stage 2
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s2_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s2_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s2_maxpool', logger=self.logger)

                # Stage 3
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[2], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s3_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s3_relu', logger=self.logger)

                # Stage 4
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[3], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s4_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s4_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s4_maxpool', logger=self.logger)

                # Stage 5
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s5_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s5_relu', logger=self.logger)

                # Stage 6
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s6_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s6_relu', logger=self.logger)

                # Stage 7
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[6], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s7_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s7_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s7_maxpool', logger=self.logger)

                # Stage 8
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[7], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s8_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s8_relu', logger=self.logger)

                # Stage 9
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[8], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s9_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s9_relu', logger=self.logger)

                # Stage 10
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[9], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s10_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s10_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s10_maxpool', logger=self.logger)

                # Stage 11
                input = tf_utils.flatten(x=input, name='s11_flatten', logger=self.logger)
                output = tf_utils.linear(input, output_size=self.conv_dims[10], initializer='xavier',
                                             name='s11_fc', logger=self.logger)
                output_relu = tf_utils.relu(input, name='s11_relu', logger=self.logger)

                # Stage 12
                score = tf_utils.linear(output_relu, output_size=self.dataset.num_class, initializer='xavier',
                                        name='s12_fc', logger=self.logger)

            elif self.method == 'VGGNet_D':
                # Stage 0
                tf_utils.print_activations(inputImg, logger=self.logger)

                # Stage 1
                input = tf_utils.conv2d(x=inputImg, output_dim=self.conv_dims[0], k_h=3, k_w=3, d_h=1, d_w=1,
                                            padding=padding, initializer='He', name='s1_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s1_relu', logger=self.logger)

                # Stage 2
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[1], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s2_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s2_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s2_maxpool', logger=self.logger)

                # Stage 3
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[2], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s3_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s3_relu', logger=self.logger)

                # Stage 4
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[3], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s4_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s4_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s4_maxpool', logger=self.logger)

                # Stage 5
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[4], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s5_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s5_relu', logger=self.logger)

                # Stage 6
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[5], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s6_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s6_relu', logger=self.logger)

                # Stage 7
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[6], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s7_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s7_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s7_maxpool', logger=self.logger)

                # Stage 8
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[7], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s8_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s8_relu', logger=self.logger)

                # Stage 9
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[8], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s9_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s9_relu', logger=self.logger)

                # Stage 10
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[9], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s10_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s10_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s10_maxpool', logger=self.logger)

                # Stage 11
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[10], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s11_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s11_relu', logger=self.logger)

                # Stage 12
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[11], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s12_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s12_relu', logger=self.logger)

                # Stage 13
                input = tf_utils.conv2d(x=input, output_dim=self.conv_dims[12], k_h=3, k_w=3, d_h=1, d_w=1,
                                        padding=padding, initializer='He', name='s13_conv', logger=self.logger)
                input = tf_utils.relu(input, name='s13_relu', logger=self.logger)

                input = tf_utils.max_pool(x=input, name='s13_maxpool', logger=self.logger)

                # Stage 14
                input = tf_utils.flatten(x=input, name='s14_flatten', logger=self.logger)
                input = tf_utils.linear(input, output_size=self.conv_dims[13], initializer='xavier',
                                         name='s14_fc', logger=self.logger)
                input = tf_utils.relu(input, name='s14_relu', logger=self.logger)

                # Stage 15
                output = tf_utils.linear(input, output_size=self.conv_dims[14], initializer='xavier',
                                         name='s15_fc', logger=self.logger)
                output_relu = tf_utils.relu(output, name='s15_relu', logger=self.logger)

                # Stage 16
                score = tf_utils.linear(output_relu, output_size=self.dataset.num_class, initializer='xavier',
                                        name='s16_fc', logger=self.logger)

            elif self.method == 'ResNet18':
                self.layers = [2, 2, 2, 2]

                # Stage 0
                tf_utils.print_activations(inputImg, logger=self.logger)

                # Stage 1
                inputs = self.conv2d_fixed_padding(inputs=inputImg, filters=64, kernel_size=7, strides=2, name='conv1')

                # Stage 2
                inputs = tf_utils.max_pool(inputs, name='3x3_maxpool', ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                           logger=self.logger)
                # Stage 3
                inputs = self.block_layer(inputs=inputs, filters=64, block_fn=self.bottleneck_block,
                                          blocks=self.layers[0],
                                          strides=1, train_mode=self.train_mode, name='block_layer1')
                # Stage 4
                inputs = self.block_layer(inputs=inputs, filters=128, block_fn=self.bottleneck_block,
                                          blocks=self.layers[1],
                                          strides=2, train_mode=self.train_mode, name='block_layer2')
                # Stage 5
                inputs = self.block_layer(inputs=inputs, filters=256, block_fn=self.bottleneck_block,
                                          blocks=self.layers[2],
                                          strides=2, train_mode=self.train_mode, name='block_layer3')
                # Stage 6
                inputs = self.block_layer(inputs=inputs, filters=512, block_fn=self.bottleneck_block,
                                          blocks=self.layers[3],
                                          strides=2, train_mode=self.train_mode, name='block_layer4')

                if self.use_batchnorm:
                    inputs = tf_utils.norm(inputs, name='before_gap_batch_norm', _type='batch', _ops=self._ops,
                                           is_train=self.train_mode, logger=self.logger)

                inputs = tf_utils.relu(inputs, name='before_flatten_relu', logger=self.logger)

                _, h, w, _ = inputs.get_shape().as_list()
                inputs = tf_utils.avg_pool(inputs, name='gap', ksize=[1, h, w, 1], strides=[1, 1, 1, 1], logger=self.logger)

                output = tf_utils.flatten(inputs, name='flatten', logger=self.logger)
                # inputs = tf_utils.linear(inputs, self.embedding_size, name='FC1')
                # inputs = tf_utils.relu(inputs, name='FC1_relu', logger=self.logger)
                #
                # output = tf_utils.linear(inputs, 32, name='FC2')
                # inputs = tf_utils.relu(output, name='FC2_relu', logger=self.logger)

                score = tf_utils.linear(output, self.dataset.num_class, name='Out')

            elif self.method == 'ResNet34':
                self.layers = [3, 4, 6, 3]

                # Stage 0
                tf_utils.print_activations(inputImg, logger=self.logger)

                # Stage 1
                inputs = self.conv2d_fixed_padding(inputs=inputImg, filters=64, kernel_size=7, strides=2, name='conv1')

                # Stage 2
                inputs = tf_utils.max_pool(inputs, name='3x3_maxpool', ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                           logger=self.logger)
                # Stage 3
                inputs = self.block_layer(inputs=inputs, filters=64, block_fn=self.bottleneck_block,
                                          blocks=self.layers[0],
                                          strides=1, train_mode=self.train_mode, name='block_layer1')
                # Stage 4
                inputs = self.block_layer(inputs=inputs, filters=128, block_fn=self.bottleneck_block,
                                          blocks=self.layers[1],
                                          strides=2, train_mode=self.train_mode, name='block_layer2')
                # Stage 5
                inputs = self.block_layer(inputs=inputs, filters=256, block_fn=self.bottleneck_block,
                                          blocks=self.layers[2],
                                          strides=2, train_mode=self.train_mode, name='block_layer3')
                # Stage 6
                inputs = self.block_layer(inputs=inputs, filters=512, block_fn=self.bottleneck_block,
                                          blocks=self.layers[3],
                                          strides=2, train_mode=self.train_mode, name='block_layer4')

                if self.use_batchnorm:
                    inputs = tf_utils.norm(inputs, name='before_gap_batch_norm', _type='batch', _ops=self._ops,
                                           is_train=self.train_mode, logger=self.logger)

                inputs = tf_utils.relu(inputs, name='before_flatten_relu', logger=self.logger)

                _, h, w, _ = inputs.get_shape().as_list()
                inputs = tf_utils.avg_pool(inputs, name='gap', ksize=[1, h, w, 1], strides=[1, 1, 1, 1], logger=self.logger)

                output = tf_utils.flatten(inputs, name='flatten', logger=self.logger)
                # inputs = tf_utils.linear(inputs, 128, name='FC1')
                # inputs = tf_utils.relu(inputs, name='FC1_relu', logger=self.logger)
                #
                # output = tf_utils.linear(inputs, 32, name='FC2')
                # inputs = tf_utils.relu(output, name='FC2_relu', logger=self.logger)

                score = tf_utils.linear(output, self.dataset.num_class, name='Out')
            return output, score

    def block_layer(self, inputs, filters, block_fn, blocks, strides, train_mode, name):
        # Only the first block per block_layer uses projection_shortcut and strides
        inputs = block_fn(inputs, filters, train_mode, self.projection_shortcut, strides, name + '_1')

        for num_iter in range(1, blocks):
            inputs = block_fn(inputs, filters, train_mode, None, 1, name=(name + '_' + str(num_iter + 1)))

        return tf.identity(inputs, name)

    def bottleneck_block(self, inputs, filters, train_mode, projection_shortcut, strides, name):
        with tf.compat.v1.variable_scope(name):
            shortcut = inputs

            if self.use_batchnorm:
                # norm(x, name, _type, _ops, is_train=True, is_print=True, logger=None)
                inputs = tf_utils.norm(inputs, name='batch_norm_0', _type='batch', _ops=self._ops,
                                       is_train=train_mode, logger=self.logger)
            inputs = tf_utils.relu(inputs, name='relu_0', logger=self.logger)

            # The projection shortcut shouldcome after the first batch norm and ReLU since it perofrms a 1x1 convolution.
            if projection_shortcut is not None:
                shortcut = self.projection_shortcut(inputs=inputs, filters_out=filters, strides=strides,
                                                    name='conv_projection')

            inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides,
                                               name='conv_0')

            if self.use_batchnorm:
                inputs = tf_utils.norm(inputs, name='batch_norm_1', _type='batch', _ops=self._ops,
                                       is_train=train_mode, logger=self.logger)
            inputs = tf_utils.relu(inputs, name='relu_1', logger=self.logger)
            inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, name='conv_1')

            output = tf.identity(inputs + shortcut, name=(name + '_output'))
            tf_utils.print_activations(output, logger=self.logger)

            return output

    def projection_shortcut(self, inputs, filters_out, strides, name):
        inputs = self.conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
                                           name=name)
        return inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, name):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)

        inputs = tf_utils.conv2d(inputs, output_dim=filters, k_h=kernel_size, k_w=kernel_size,
                                 d_h=strides, d_w=strides, initializer='He', name=name,
                                 padding=('SAME' if strides == 1 else 'VALID'), logger=self.logger)
        return inputs

    def _loss(self, is_siamese=True):
        # Siamese loss
        if is_siamese:
            true_label = self.y_
            false_label = tf.subtract(1.0, self.y_)

            M = tf.constant(self.margin)
            distance_pos = tf.pow(tf.subtract(self.gw1, self.gw2), 2)  # tf.subtract(self.gw1, self.gw2)
            distance_pos = tf.reduce_sum(distance_pos, 1)

            distance_neg = distance_pos
            distance_neg = tf.subtract(M, tf.sqrt(distance_neg + 1e-6))
            distance_neg = tf.maximum(distance_neg, 0)
            distance_neg = tf.pow(distance_neg, 2)

            pos = tf.multiply(true_label, distance_pos)  # distance2
            neg = tf.multiply(false_label, distance_neg)  # tf.pow(tf.maximum(tf.subtract(M, distance), 0), 2))

            self.siamese_loss = tf.add(pos, neg)
            self.siamese_loss = self.flags.lambda_1 * tf.math.reduce_mean(self.siamese_loss)

        else:
            self.siamese_loss = 0

        # Classification loss
        if is_siamese:
            self.cls_data_loss_1 = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_1, logits=self.score1))
            self.cls_data_loss_2 = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_2, logits=self.score2))
        else:
            self.cls_data_loss_1 = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_1, logits=self.score1))
            self.cls_data_loss_2 = 0

        # Regularization term
        variables = self.get_regularization_variables()
        self.reg_term = self.weight_decay * tf.math.reduce_mean([tf.nn.l2_loss(variable) for variable in variables])

        # self.total_loss = self.data_loss + self.reg_term
        self.total_loss = self.cls_data_loss_1 + self.cls_data_loss_2 + self.siamese_loss + self.reg_term

        return self.total_loss

    def _tensorboard(self):
        if self.is_train:
            self.tb_total = tf.compat.v1.summary.scalar('loss/total_loss', self.total_loss)
            self.tb_siamese = tf.compat.v1.summary.scalar('loss/siamese_loss', self.siamese_loss)
            self.tb_reg = tf.compat.v1.summary.scalar('loss/reg_term', self.reg_term)
            self.tb_cls_data1 = tf.compat.v1.summary.scalar('loss/cls_data_loss_1', self.cls_data_loss_1)
            self.tb_cls_data2 = tf.compat.v1.summary.scalar('loss/cls_data_loss_2', self.cls_data_loss_2)
            self.tb_accuracy = tf.compat.v1.summary.scalar('loss/accuracy', self.accuracy)

            self.summary_op = tf.summary.merge_all()

        # if self.is_train:
        #     self.tb_total = tf.compat.v1.summary.scalar('loss/total_loss', self.total_loss)
        #     self.tb_data = tf.compat.v1.summary.scalar('loss/data_loss', self.data_loss)
        #     self.tb_reg = tf.compat.v1.summary.scalar('loss/reg_term', self.reg_term)
        #     self.tb_batch_acc = tf.compat.v1.summary.scalar('Acc/batch_acc', self.batch_acc)
        #     self.summary_op = tf.compat.v1.summary.merge(
        #         inputs=[self.tb_total, self.tb_data, self.tb_reg, self.tb_lr, self.tb_batch_acc])
        #
        # self.tb_accuracy = tf.compat.v1.summary.scalar('acc/val_acc', self.accuracy_metric * 100.)
        # self.metric_summary_op = tf.compat.v1.summary.merge(inputs=[self.tb_accuracy])

    def print_info(self, total_loss, siamese_loss, reg_term, cls_loss_1, cls_loss_2, accuracy, iter_time):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            # self.Calculate_accuracy()
            ord_output = collections.OrderedDict([('cur_iter', iter_time),
                                                  ('total_iters', self.flags.iters),
                                                  ('batch_size', self.flags.batch_size),
                                                  ('total_loss', total_loss),
                                                  (' - siamese_loss', siamese_loss),
                                                  (' - reg_loss', reg_term),
                                                  (' - cls_loss1', cls_loss_1),
                                                  (' - cls_loss2', cls_loss_2),
                                                  ('accuracy', accuracy),
                                                  ('dataset', self.flags.dataset),
                                                  ('gpu_index', self.flags.gpu_index)])

            utils.print_metrics(iter_time, ord_output)

    def sample_imgs(self, iter_time, sample_out_dir):
        x_test, y_test = self.dataset.test_sample()
        feed = {self.x1: x_test,
                self.train_mode: False}
        coordinates = self.sess.run(self.gw1, feed_dict=feed)
        if self.method == 'cifar10_v4' or 'fingervein' or 'fingervein_v2' or 'ResNet18' or 'ResNet34':
            tsne = TSNE(n_components=2)
            coordinates = tsne.fit_transform(coordinates)

        self.Visualize(iter_time, sample_out_dir, coordinates, y_test)

    def train_sample_imgs(self, iter_time, sample_out_dir, x1_imgs, x1_label):
        # train part
        total_train_feature_map = np.empty((x1_imgs.shape[0], self.embedding_size), dtype=np.float32)
        cnt = 0

        while (cnt * self.flags.batch_size < x1_imgs.shape[0]):
            start_iter = cnt * self.flags.batch_size
            end_iter = min((cnt + 1) * self.flags.batch_size, x1_imgs.shape[0])

            feed_train = {self.x1: x1_imgs[start_iter:end_iter],
                          self.train_mode: True}
            coordinates_train = self.sess.run(self.gw1, feed_dict=feed_train)

            total_train_feature_map[start_iter:end_iter, :] = coordinates_train
            cnt += 1

        # val part
        x_test, y_test = self.dataset.test_data, self.dataset.test_label
        total_feature_map = np.empty((x_test.shape[0], self.embedding_size), dtype=np.float32)#10000
        cnt = 0

        # total_pred_cls = np.empty((10000,), dtype=np.uint8)

        while(cnt * self.flags.batch_size < x_test.shape[0]):#10000
            start_iter = cnt*self.flags.batch_size
            end_iter = min((cnt+1)*self.flags.batch_size, x_test.shape[0])#10000

            feed_test = {self.x1: x_test[start_iter:end_iter],
                         self.train_mode: False}
            coordinates_test = self.sess.run(self.gw1, feed_dict=feed_test)
            # coordinates_test, pred_cls = self.sess.run([self.gw1, self.pred_cls], feed_dict=feed_test)

            total_feature_map[start_iter:end_iter, :] = coordinates_test
            # total_pred_cls[start_iter:end_iter] = pred_cls
            cnt += 1

        tsne = TSNE(n_components=2)
        total_test_coordinates = tsne.fit_transform(total_feature_map)
        total_train_coordinates = tsne.fit_transform(total_train_feature_map)

        self.Visualize(iter_time, sample_out_dir, total_train_coordinates, x1_label, total_test_coordinates, y_test)

    def Calculate_accuracy(self):
        ##############################################################################################################
        x_train, y_train = self.dataset.train_data, self.dataset.train_label
        cnt = 0

        total_pred_cls = np.empty((x_train.shape[0],), dtype=np.int32)

        #print("Calculate accuracy for train...")
        while cnt * self.flags.batch_size < x_train.shape[0]:
            print("{} ".format(cnt), end='')
            start_iter = cnt * self.flags.batch_size
            end_iter = min((cnt + 1) * self.flags.batch_size, x_train.shape[0])

            feed_train = {self.x1: x_train[start_iter:end_iter],
                          self.train_mode: True}
            pred_cls = self.sess.run(self.pred_cls, feed_dict=feed_train)

            total_pred_cls[start_iter:end_iter] = pred_cls
            cnt += 1

        print()
        self.train_accuracy = np.mean(np.equal(y_train, total_pred_cls))
        print("train_accuracy:{}".format(self.train_accuracy))


        ##############################################################################################################
        x_test, y_test = self.dataset.test_data, self.dataset.test_label# self.dataset.test_sample(self.num_test)
        cnt = 0

        total_pred_cls = np.empty((x_test.shape[0],), dtype=np.int32)

        print("Calculate accuracy for test data...")
        while cnt * self.flags.batch_size < x_test.shape[0]:
            print("{} ".format(cnt), end='')
            start_iter = cnt * self.flags.batch_size
            end_iter = min((cnt + 1) * self.flags.batch_size, x_test.shape[0])

            feed_test = {self.x1: x_test[start_iter:end_iter],
                         self.train_mode: False}
            pred_cls = self.sess.run(self.pred_cls, feed_dict=feed_test)

            total_pred_cls[start_iter:end_iter] = pred_cls
            cnt += 1
        print()
        # print("\ny_test:{}, total_pred_cls:{}, same?:{}".format(y_test[0], total_pred_cls[0],
        #                                                         np.equal(y_test[0], total_pred_cls[0])))
        # print("\ny_test:{}, total_pred_cls:{}, same?:{}".format(y_test[1], total_pred_cls[1],
        #                                                         np.equal(y_test[1], total_pred_cls[1])))
        # print("\ny_test:{}, total_pred_cls:{}, same?:{}".format(y_test[2], total_pred_cls[2],
        #                                                         np.equal(y_test[2], total_pred_cls[2])))
        # print("\ny_test:{}, total_pred_cls:{}, same?:{}".format(y_test[3], total_pred_cls[3],
        #                                                         np.equal(y_test[3], total_pred_cls[3])))
        # print("\ny_test:{}, total_pred_cls:{}, same?:{}".format(y_test[4], total_pred_cls[4],
        #                                                         np.equal(y_test[4], total_pred_cls[4])))
        # print("\ny_test:{}, total_pred_cls:{}, same?:{}".format(y_test[5], total_pred_cls[5],
        #                                                         np.equal(y_test[5], total_pred_cls[5])))
        self.accuracy = np.mean(np.equal(y_test, total_pred_cls))
        print("test_accuracy:{}".format(self.accuracy))

    def Calculate_test_accuracy(self):
        x_test, y_test = self.dataset.test_data, self.dataset.test_label# self.dataset.test_sample(self.num_test)
        print("x_test.shape[0]:{}".format(x_test.shape[0]))
        print("y_test.shape[0]:{}".format(y_test.shape[0]))
        cnt = 0

        total_pred_cls = np.empty((x_test.shape[0],), dtype=np.int32)

        print("Calculate accuracy for test data...")
        while cnt * self.flags.batch_size < x_test.shape[0]:
            print("{} ".format(cnt), end='')
            start_iter = cnt * self.flags.batch_size
            end_iter = min((cnt + 1) * self.flags.batch_size, x_test.shape[0])

            feed_test = {self.x1: x_test[start_iter:end_iter],
                         self.train_mode: False}
            pred_cls = self.sess.run(self.pred_cls, feed_dict=feed_test)

            total_pred_cls[start_iter:end_iter] = pred_cls
            print("pred_cls:{}".format(pred_cls))
            cnt += 1

        print("\ny_test:{}".format(y_test))
        print("y_test.shape[0]:{}".format(y_test.shape[0]))
        print("total_pred_cls:{}".format(total_pred_cls))
        print("total_pred_cls.shape[0]:{}".format(total_pred_cls.shape[0]))
        self.accuracy = np.mean(np.equal(y_test, total_pred_cls))
        print("\ntest accuracy : {}".format(self.accuracy))

    def Visualize(self, iter_time, sample_out_dir, coordinates_train, y_train, coordinates_test=None, y_test=None):
        if coordinates_test is not None:
            plt.figure(figsize=(12, 6))
            sub1 = plt.subplot(121)
        else:
            plt.figure(figsize=(6, 6))
            sub1 = plt.subplot(111)

        colormap = plt.get_cmap('hsv')  # tab10

        # train part
        feat = coordinates_train
        sub1_min = np.min(coordinates_train, 0)
        sub1_max = np.max(coordinates_train, 0)

        # list_ = list(set(y_train)).sort()
        rand_class = np.random.choice(self.dataset.num_class, size=10, replace=False)
        rand_list = list(rand_class)
        rand_list.sort()
        print("rand_class:{}".format(rand_class))
        for i in range(feat.shape[0]):
            if i//self.dataset.num_sample_each_class not in rand_class:
                continue
            dot = plt.scatter(feat[i, 0], feat[i, 1], alpha=1, s=10,
                              color=colormap(rand_list.index(y_train[i]) / 10, )[:3])  # list_.index(y_train[i])/len(list_), y_test[i]/ 10.
                              # color=colormap(y_train[i]/self.dataset.num_class,)[:3])#list_.index(y_train[i])/len(list_), y_test[i]/ 10.
            sub1.add_artist(dot)#imagebox

        plt.axis([sub1_min[0], sub1_max[0], sub1_min[1], sub1_max[1]])
        if iter_time is 'text':
            plt.title('Result from the test set')
        else:
            plt.title('Result from the {}th iter times'.format(iter_time))

        # test part
        if coordinates_test is not None:
            feat2 = coordinates_test
            sub2_min = np.min(coordinates_test, 0)
            sub2_max = np.max(coordinates_test, 0)

            sub2 = plt.subplot(122)

            for i in range(feat2.shape[0]):
                dot = plt.scatter(feat2[i, 0], feat2[i, 1], alpha=0.1, s=10,
                                  color=colormap(y_test[i]/self.dataset.num_class,)[:3])#list_.index(y_train[i])/len(list_), y_test[i]/ 10.
                sub2.add_artist(dot)

            plt.axis([sub2_min[0], sub2_max[0], sub2_min[1], sub2_max[1]])
            plt.title('Result from the test set')

        plt.savefig(os.path.join(sample_out_dir, str(iter_time)))

    def draw_histogram(self, test_out_dir):
        feature_1800x512 = np.empty((self.dataset.test_data.shape[0], 512), dtype=float)
        cnt = 0

        print("Draw histogram for test data...")
        while cnt * self.flags.batch_size < self.dataset.test_data.shape[0]:
            print("{} ".format(cnt), end='')
            start_iter = cnt * self.flags.batch_size
            end_iter = min((cnt + 1) * self.flags.batch_size, self.dataset.test_data.shape[0])

            feed_test = {self.x1: self.dataset.test_data[start_iter:end_iter],
                         self.train_mode: False}
            feature_1800x512[start_iter:end_iter] = self.sess.run(self.gw1, feed_dict=feed_test)
            cnt += 1
        # feed = {self.x1: self.dataset.test_data}
        # feature_1800x512 = self.sess.run(self.gw1, feed_dict=feed)

        num_data_each_class = self.dataset.num_sample_each_class \
                                   - int(self.dataset.num_sample_each_class * self.dataset.train_rate)

        genuine_match = np.empty((int(self.dataset.num_class * (num_data_each_class * (num_data_each_class-1))/2), ))  # 600 * 3 * 2 / 2
        for i in range(self.dataset.num_class):
            for j in range(num_data_each_class):
                genuine_match[i * num_data_each_class + j] = np.sqrt(np.sum((feature_1800x512[i * num_data_each_class + j]
                            - feature_1800x512[i * num_data_each_class + ((j+1) % num_data_each_class)]) ** 2))

        imposter_match = np.empty((int((self.dataset.num_class * (self.dataset.num_class - 1)) / 2 * num_data_each_class
                                   * num_data_each_class), ))  # 600 * 599 / 2 * 3 * 3

        img_list = [i for i in range(feature_1800x512.shape[0])]
        cnt = 0
        for i in range(600):
            imposter_1 = []
            for idx in range(num_data_each_class):
                imposter_1.append(i * num_data_each_class + idx)

            img_list = list(set(img_list) - set(imposter_1))
            print("img_list:{}".format(img_list))
            for j in imposter_1:
                for k in img_list:
                    print("{}th imposter match:{} - {}".format(cnt, j, k))
                    imposter_match[cnt] = np.sqrt(np.sum((feature_1800x512[j] - feature_1800x512[k]) ** 2))
                    cnt += 1

        max_genu_impo = max(np.max(genuine_match), np.max(imposter_match))
        genuine_match /= max_genu_impo
        imposter_match /= max_genu_impo
        print("max_genu_impo:{}".format(max_genu_impo))

        n, bins, patches = plt.hist(x=imposter_match, bins='auto', range=(0, 1), color='red', alpha=0.5, normed=True)  # / imposter_match.shape[0]  # , rwidth=0.85, alpha=0.75
        if self.flags.threshold:
            FAR = 0
            impo_x_distance = bins[1] - bins[0]
            for i in range(len(n)):
                # print("n[i]:{}, bins[i]:{}".format(n[i], bins[i]))
                if bins[i] < self.flags.threshold:
                    FAR += n[i]

        n, bins, patches = plt.hist(x=genuine_match, bins=bins, range=(0, 1), color='blue', alpha=0.5, normed=True)  # / genuine_match.shape[0]rwidth=0.85,
        if self.flags.threshold:
            FRR = 0
            genu_x_distance = bins[1] - bins[0]
            for i in range(len(n)):
                # print("n[i]:{}, bins[i]:{}".format(n[i], bins[i]))
                if bins[i] > self.flags.threshold:
                    FRR += n[i]

            print("FAR:{}, impo_x_distance:{}".format(FAR, impo_x_distance))
            print("FRR:{}, genu_x_distance:{}".format(FRR, genu_x_distance))

        plt.grid(axis='y', )  # alpha=0.75
        plt.xlabel('Normalized Matching Distance')
        plt.ylabel('Frequency')
        plt.title('Genuine & Imposter Histogram')  # Poster
        plt.legend(['Imposter', 'Genuine'])

        plt.savefig(os.path.join(test_out_dir, 'Histogram'))
        plt.show()

    def save_features(self, test_out_dir):
        features = np.empty((self.dataset.test_data.shape[0], 512), dtype=float)
        cnt = 0

        print("Save features for test data...")
        while cnt * self.flags.batch_size < self.dataset.test_data.shape[0]:
            print("{} ".format(cnt), end='')
            start_iter = cnt * self.flags.batch_size
            end_iter = min((cnt + 1) * self.flags.batch_size, self.dataset.test_data.shape[0])

            feed_test = {self.x1: self.dataset.test_data[start_iter:end_iter],
                         self.train_mode: False}
            features[start_iter:end_iter] = self.sess.run(self.gw1, feed_dict=feed_test)
            cnt += 1

        np.save(os.path.join(test_out_dir, self.flags.load_model+'_feature'), features)

    @staticmethod
    def set_conv_dims(method):
        conv_dims = None

        if method == 'mnist':
            conv_dims = [1024, 1024, 2]

        elif method == 'cifar10_v4':
            conv_dims = [64, 128, 256, 256, 256, 256, 256, 256, 256, 128, 64, 32, 10]
            # conv_dims = [64, 128, 256, 256, 256, 256, 256, 256, 256, 128, 64, 32, 10]
            # conv_dims = [64, 128, 256, 256, 256, 256, 256, 256, 256, 256, 128, 64, 10]
            # conv_dims = [64, 128, 256, 256, 256, 256, 256, 256, 1024, 512, 256, 128, 10]
            # conv_dims = [64, 128, 256, 256, 256, 256, 256, 256, 1024, 1024, 512, 256, 10]
            # conv_dims = [64, 128, 256, 256, 256, 256, 256, 256, 1024, 1024, 1024, 512, 10]
            # conv_dims = [64, 128, 256, 256, 256, 256, 256, 256, 1024, 1024, 1024, 1024, 10]

        elif method == 'fingervein':
            conv_dims = [128, 256, 512, 768, 1024]

        elif method == 'fingervein_v2':
            conv_dims = [64, 64, 128, 128, 256, 256, 256, 256, 256, 256, 1024]

        elif method == 'VGGNet_D':
            conv_dims = [64, 64, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 4096, 1024]

        elif method == 'ResNet18':
            conv_dims = None

        elif method == 'ResNet34':
            conv_dims = None

        else:
            exit(" [!]Cannot find the defined method {} !".format(method))

        return conv_dims

    @staticmethod
    def get_regularization_variables():
        # We exclude 'bias', 'beta' and 'gamma' in batch normalization
        variables = [variable for variable in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
                     if ('bias' not in variable.name) and
                     ('beta' not in variable.name) and
                     ('gamma' not in variable.name)]

        return variables

    @staticmethod
    def fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_start = pad_total // 2
        pad_end = pad_total - pad_start
        inputs = tf.pad(inputs, [[0, 0], [pad_start, pad_end], [pad_start, pad_end], [0, 0]])
        return inputs