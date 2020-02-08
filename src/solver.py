# --------------------------------------------------------
# Tensorflow Implementation of Siamese Network
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------

import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

from dataset import Dataset
from model import Siamese
import tensorflow_utils as tf_utils

class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=run_config)

        self.flags = flags
        self.dataset = Dataset(self.flags.dataset, is_train=self.flags.is_train)
        self.model = Siamese(self.sess, self.flags, self.dataset.image_size, self.dataset)
        self.accuracy = self.model.accuracy
        self.train_accuracy = self.model.train_accuracy ###############################################################

        self._make_folders()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        tf_utils.show_all_variables()

    def _make_folders(self):
        if self.flags.is_train:
            cur_time = datetime.now().strftime("%Y%m%d-%H%M")
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.model_out_dir):
                os.makedirs(self.model_out_dir)

            self.sample_out_dir = "{}/sample/{}".format(self.flags.dataset, cur_time)
            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)

            self.train_writer = tf.summary.FileWriter("{}/logs/{}".format(self.flags.dataset, cur_time),
                                                      graph_def=self.sess.graph_def)
        else:
            self.model_out_dir = "{}/model/{}".format(self.flags.dataset, self.flags.load_model)
            self.test_out_dir = "{}/test/{}".format(self.flags.dataset, self.flags.load_model)
            if not os.path.isdir(self.test_out_dir):
                os.makedirs(self.test_out_dir)

    def train(self):
        for iter_time in range(self.flags.iters):
            if self.flags.is_siamese:
                batch_imgs1, batch_label1, batch_imgs2, batch_label2 = self.dataset.train_next_batch_pair(
                    batch_size=self.flags.batch_size)

            else:
                batch_imgs1, batch_label1 = self.dataset.train_next_batch_random(batch_size=self.flags.batch_size)
                batch_imgs2 = None
                batch_label2 = None

            total_loss, siamese_loss, reg_term, cls_loss_1, cls_loss_2, summary = self.model.train_step(
                batch_imgs1, batch_label1, batch_imgs2, batch_label2, is_siamese=self.flags.is_siamese)

            self.model.print_info(total_loss, siamese_loss, reg_term, cls_loss_1, cls_loss_2, self.model.accuracy, iter_time)

            if iter_time % self.flags.eval_freq == 0:
                print("Evaluaton process...")
                self.model.Calculate_accuracy()

            self.train_writer.add_summary(summary, iter_time)
            self.train_writer.flush()

            # self.train_sample(iter_time, batch_imgs1, batch_label1)
            # self.train_sample(iter_time, self.dataset.train_data, self.dataset.train_label)
            # self.train_sample(iter_time, self.dataset.train_data.images, self.dataset.train_data.labels)

            # save model
            self.save_model(iter_time)

        self.save_model(self.flags.iters)

    def test(self):
        if self.load_model():
            print(' [*] Load SUCCESS!')
        else:
            print(' [!] Load Failed...')

        num_iters = 1
        total_time = 0.
        for iter_time in range(num_iters):
            # measure inference time
            start_time = time.time()

            ################################################
            # self.model.draw_histogram(self.test_out_dir)
            # self.model.save_features(self.test_out_dir)
            # self.model.train_sample_imgs(iter_time, self.test_out_dir, self.dataset.train_data, self.dataset.train_label)
            self.model.Calculate_test_accuracy()
            ################################################
            total_time += time.time() - start_time


        print('Avg PT: {:.2f} msec.'.format(total_time / num_iters * 1000.))


    def save_model(self, iter_time):
        # print('self.train_accuracy:{}, self.model.train_accuracy:{}'.format(self.train_accuracy, self.model.train_accuracy))
        if self.train_accuracy < self.model.train_accuracy:
            self.train_accuracy = self.model.train_accuracy
        print('self.accuracy:{}, self.model.accuracy:{}\n'.format(self.accuracy, self.model.accuracy))
        if np.mod(iter_time + 1, self.flags.save_freq) == 0 and self.accuracy < self.model.accuracy:
            model_name = 'model'
            self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name), global_step=iter_time)
            self.accuracy = self.model.accuracy

            print('=====================================')
            print('             Model saved!            ')
            print('=====================================\n')

    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            self.model.sample_imgs(iter_time, self.sample_out_dir)
            # self.model.plots(imgs, iter_time, self.sample_out_dir)

    def train_sample(self, iter_time, x1_imgs, x1_label):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            self.model.train_sample_imgs(iter_time, self.sample_out_dir, x1_imgs, x1_label)
            # self.model.plots(imgs, iter_time, self.sample_out_dir)

    # def train_all_sample(self, iter_time, train_data, train_label):
    #     if np.mod(iter_time, self.flags.sample_freq) == 0:
    #         self.model.train_sample_imgs(iter_time, self.sample_out_dir, train_data, train_label)
    #         # self.model.plots(imgs, iter_time, self.sample_out_dir)

    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # print("ckpt_name:{}".format(ckpt_name))
            # print("os.path.join(self.model_out_dir, ckpt_name):{}".format(os.path.join(self.model_out_dir, ckpt_name)))
            # self.saver = tf.train.import_meta_graph('{}.meta'.format(os.path.join(self.model_out_dir, ckpt_name)))
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            meta_graph_path = ckpt.model_checkpoint_path + '.meta'
            self.iter_time = int(meta_graph_path.split('-')[-1].split('.')[0])

            print(' [*] Load iter_time: {}'.format(self.iter_time))

            return True
        else:
            return False

