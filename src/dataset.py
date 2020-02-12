# --------------------------------------------------------
# Tensorflow Implementation of Siamese Network
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

import time

class MnistDataset(object):
    def __init__(self, dataset_name, is_train=True):
        self.dataset_name = dataset_name
        self.image_size = (28, 28, 1)
        self.num_trains = 0
        self.num_class = 10

        self.mnist_path = os.path.join('../Dataset', self.dataset_name)
        self._load_mnist()

    def _load_mnist(self):
        print('Load {} dataset...'.format(self.dataset_name))
        self.load_train_data = input_data.read_data_sets(self.mnist_path, one_hot=False)
        self.train_data = self.load_train_data.train.images.reshape([-1, *self.image_size])
        self.train_label = self.load_train_data.train.labels
        self.num_trains = self.load_train_data.train.num_examples

        self.load_test_data = input_data.read_data_sets(self.mnist_path, one_hot=False)

        self.test_data = self.load_test_data.test.images.reshape([-1, *self.image_size])
        self.test_label = self.load_test_data.test.labels
        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch_random(self, batch_size=512):
        batch_imgs, batch_label = self.load_train_data.train.next_batch(batch_size)

        # reshape 784 vector to 28 x 28 x 1
        batch_imgs = np.reshape(batch_imgs, [batch_size, *self.image_size])

        return batch_imgs, batch_label

    def train_next_batch_pair(self, batch_size=512):
        self.half_batch_size = int(np.ceil(batch_size/2))
        self.labels_set = set(self.train_label)

        self.label_to_indices = {label:np.where(self.train_label == label)[0] for label in self.labels_set}

        rand_positive_num = np.random.choice(self.num_trains, self.half_batch_size, replace=False)
        rand_negative_num = np.random.choice(self.num_trains, batch_size - self.half_batch_size, replace=False)

        positive_pairs = [[i, np.random.choice(self.label_to_indices[self.train_label[i].item()])]
                          for i in rand_positive_num]

        negative_pairs = [[i, np.random.choice(self.label_to_indices[np.random.choice(list(self.labels_set - set(
            [self.train_label[i].item()])))])] for i in rand_negative_num]

        list1 = []
        list2 = []
        for i in range(self.half_batch_size):
            list1.append(positive_pairs[i][0])
            list2.append(positive_pairs[i][1])

        for i in range(batch_size - self.half_batch_size):
            list1.append(negative_pairs[i][0])
            list2.append(negative_pairs[i][1])

        batch_imgs1 = self.train_data[list1]
        batch_label1 = self.train_label[list1]

        batch_imgs2 = self.train_data[list2]
        batch_label2 = self.train_label[list2]

        return batch_imgs1, batch_label1, batch_imgs2, batch_label2

    def test_sample(self):
        test_data = self.load_test_data.test.images
        test_label = self.load_test_data.test.labels
        # reshape 784 vector to 28 x 28 x 1
        test_data = np.reshape(test_data, [self.test_data.test.num_examples, *self.image_size])

        return test_data, test_label


class Cifar10(object):
    def __init__(self, dataset_name, is_train=True):
        self.dataset_name = dataset_name
        self.image_size = (32, 32, 3)
        self.num_trains = 0
        self.num_class = 10

        self.cifar10_path = os.path.join('../Dataset', self.dataset_name)
        self._load_cifar10()

    def _load_cifar10(self):
        import cifar10

        cifar10.data_path = self.cifar10_path
        print('Load {} dataset...'.format(self.dataset_name))

        # The CIFAR-10 data-set is about 163 MB and will be downloaded automatically if it is not
        # located in the given path.
        cifar10.maybe_download_and_extract()

        self.train_data, self.train_label, _ = cifar10.load_training_data()
        self.num_trains = self.train_data.shape[0]
        self.test_data, self.test_label, _ = cifar10.load_test_data()
        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def train_next_batch_random(self, batch_size=512):
        rand = np.random.choice(self.num_trains, batch_size, replace=False)

        batch_imgs = self.train_data[rand]
        batch_label = self.train_label[rand]

        return batch_imgs, batch_label

    def train_next_batch_pair(self, batch_size=512):
        self.half_batch_size = int(np.ceil(batch_size/2))
        self.labels_set = set(self.train_label)

        self.label_to_indices = {label:np.where(self.train_label == label)[0] for label in self.labels_set}

        rand_positive_num = np.random.choice(self.num_trains, self.half_batch_size, replace=False)
        rand_negative_num = np.random.choice(self.num_trains, batch_size - self.half_batch_size, replace=False)

        positive_pairs = [[i, np.random.choice(self.label_to_indices[self.train_label[i].item()])]
                          for i in rand_positive_num]

        negative_pairs = [[i, np.random.choice(self.label_to_indices[np.random.choice(list(self.labels_set - set(
            [self.train_label[i].item()])))])] for i in rand_negative_num]

        list1 = []
        list2 = []
        for i in range(self.half_batch_size):
            list1.append(positive_pairs[i][0])
            list2.append(positive_pairs[i][1])

        for i in range(batch_size - self.half_batch_size):
            list1.append(negative_pairs[i][0])
            list2.append(negative_pairs[i][1])

        batch_imgs1 = self.train_data[list1]
        batch_label1 = self.train_label[list1]

        batch_imgs2 = self.train_data[list2]
        batch_label2 = self.train_label[list2]

        return batch_imgs1, batch_label1, batch_imgs2, batch_label2

    def test_sample(self, batch_size=512):
        indexes = np.random.randint(low=0, high=10000, size=batch_size)
        test_data, test_label = self.test_data[indexes], self.test_label[indexes]

        # test_data, test_label = self.test_data, self.test_label

        # reshape 784 vector to 32 x 32 x 3
        test_data = np.reshape(test_data, [test_data.shape[0], *self.image_size])
        return test_data, test_label


class FingerVein(object):
    def __init__(self, dataset_name, is_train=True):
        self.dataset_name = dataset_name
        self.image_size = (120, 300, 1)  #(240, 600, 1)
        self.num_trains = 0
        self.num_class = 600
        self.is_train = is_train

        self.num_sample_each_class = 10
        self.train_rate = 0.7
        self.test_sample_num_list = [8, 4, 5]#[0, 2, 3]#

        self.fingervein_path = os.path.join('../Dataset', self.dataset_name)
        self._load_fingervein_path()

    def _load_fingervein_path(self):
        self.data_path = self.fingervein_path
        print('Load {} dataset...'.format(self.dataset_name))
        if self.is_train:
            self.train_data, self.train_label, self.test_data, self.test_label = self.load_training_and_test_data()
            self.num_trains = len(self.train_data)

        else:
            # self.test_data, self.test_label = self.load_Specific_test_data(self.test_sample_num_list)

            # _, _, _, _, = self.load_training_and_test_data()
            # self.train_data, self.train_label, self.test_data, self.test_label\
            #     = self.total_data, self.total_label, self.total_data, self.total_label

            self.train_data, self.train_label, self.test_data, self.test_label = self.load_training_and_test_data()
            self.num_trains = len(self.train_data)

        print('Load {} dataset SUCCESS!'.format(self.dataset_name))

    def load_training_and_test_data(self):
        import cv2
        file_list = os.listdir(self.data_path)
        file_list.sort()

        self.total_data = np.zeros([len(file_list), *self.image_size], dtype=float)
        self.total_label = np.zeros([len(file_list), ], dtype=int)

        for i, file_name in enumerate(file_list):
            # print(i, file_name)
            img_name = os.path.join(self.data_path, file_name)
            img = cv2.imread(img_name, 0)

            self.total_data[i, :, :] = img.reshape(-1, *self.image_size)
            self.total_label[i] = i//self.num_sample_each_class

        # rand_class = np.random.choice(self.num_class, size=self.use_num_class, replace=False)
        num_of_train = int(self.num_sample_each_class * self.train_rate) # each class

        total_sample_num = np.array(range(self.num_sample_each_class))
        if self.is_train:
            train_sample_num = np.random.choice(self.num_sample_each_class, num_of_train, replace=False)
            test_sample_num = np.array(list(set(total_sample_num) - set(train_sample_num)))

        else:
            test_sample_num = np.array(self.test_sample_num_list)
            train_sample_num = np.array(list(set(total_sample_num) - set(test_sample_num)))

        print("train_sample_num:{}".format(train_sample_num))
        print("test_sample_num:{}".format(test_sample_num))

        train_idx = list(cls*self.num_sample_each_class + num_ for cls in range(self.num_class) for num_ in train_sample_num)
        test_idx = list(cls*self.num_sample_each_class + num_ for cls in range(self.num_class) for num_ in test_sample_num)

        train_data = self.total_data[train_idx, :, :]#np.asarray(train_data_list)
        train_data = train_data.reshape(-1, *self.image_size)

        train_label = self.total_label[train_idx]#np.asarray(train_label_list)
        train_label = train_label.reshape(-1,)

        test_data = self.total_data[test_idx, :, :]#np.asarray(test_data_list)
        test_data = test_data.reshape(-1, *self.image_size)

        test_label = self.total_label[test_idx]#np.asarray(test_label_list)
        test_label = test_label.reshape(-1, )

        return train_data, train_label, test_data, test_label

    def load_Specific_test_data(self, test_sample_num_list):
        import cv2
        file_list = os.listdir(self.data_path)
        file_list.sort()

        self.total_data = np.zeros([len(file_list), *self.image_size], dtype=float)
        self.total_label = np.zeros([len(file_list), ], dtype=int)

        for i, file_name in enumerate(file_list):
            # print(i, file_name)
            img_name = os.path.join(self.data_path, file_name)
            img = cv2.imread(img_name, 0)

            self.total_data[i, :, :] = img.reshape(-1, *self.image_size)
            self.total_label[i] = i // self.num_sample_each_class

        test_sample_num = np.array(test_sample_num_list)

        test_idx = list(
            cls * self.num_sample_each_class + num_ for cls in range(self.num_class) for num_ in test_sample_num)
        print("len(test_idx):{}".format(len(test_idx)))
        test_data = self.total_data[test_idx, :, :]
        test_data = test_data.reshape(-1, *self.image_size)

        test_label = self.total_label[test_idx]
        test_label = test_label.reshape(-1, )

        return test_data, test_label

    def train_next_batch_random(self, batch_size=512):
        rand = np.random.choice(self.num_trains, batch_size, replace=False)

        batch_imgs = self.train_data[rand]
        batch_label = self.train_label[rand]

        return batch_imgs, batch_label

    def train_next_batch_pair(self, batch_size=512):
        self.half_batch_size = int(np.ceil(batch_size/2))
        self.labels_set = set(self.train_label)

        self.label_to_indices = {label:np.where(self.train_label == label)[0] for label in self.labels_set}

        rand_positive_num = np.random.choice(self.num_trains, self.half_batch_size, replace=True)
        rand_negative_num = np.random.choice(self.num_trains, batch_size - self.half_batch_size, replace=True)

        positive_pairs = [[i, np.random.choice(list(set(self.label_to_indices[self.train_label[i].item()]) - set([i])))]
                          for i in rand_positive_num]

        negative_pairs = [[i, np.random.choice(self.label_to_indices[np.random.choice(
            list(self.labels_set - set([self.train_label[i].item()])))])] for i in rand_negative_num]

        list1 = []
        list2 = []
        for i in range(self.half_batch_size):
            list1.append(positive_pairs[i][0])
            list2.append(positive_pairs[i][1])

        for i in range(batch_size - self.half_batch_size):
            list1.append(negative_pairs[i][0])
            list2.append(negative_pairs[i][1])

        batch_imgs1 = self.train_data[list1]
        batch_label1 = self.train_label[list1]

        batch_imgs2 = self.train_data[list2]
        batch_label2 = self.train_label[list2]

        return batch_imgs1, batch_label1, batch_imgs2, batch_label2

    def test_sample(self, batch_size=512):
        indexes = np.random.randint(low=0, high=self.test_data.shape[0], size=batch_size)
        test_data, test_label = self.test_data[indexes], self.test_label[indexes]

        # test_data, test_label = self.test_data, self.test_label

        # reshape 784 vector to 32 x 32 x 3
        test_data = np.reshape(test_data, [test_data.shape[0], *self.image_size])
        return test_data, test_label

# noinspection PyPep8Naming
def Dataset(dataset_name, is_train=True):
    if dataset_name == 'mnist':
        return MnistDataset(dataset_name, is_train)
    elif dataset_name == 'cifar10':
        return Cifar10(dataset_name, is_train)
    elif dataset_name == 'fingervein':
        return FingerVein(dataset_name, is_train)
    else:
        raise NotImplementedError

    # tf.logging.set_verbosity(old_v)
