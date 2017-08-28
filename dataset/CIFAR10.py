'''Author: Yu Wang(wangyu@in.tum.de)
   LICENSE: Open source.
   This script processes row CIFAR10 dataset into desired batched images.
   '''
from __future__ import print_function

import os
import sys
import random
import numpy as np
import cPickle
import glob

class CIFAR10():
    '''CIFAR10 class, includes all necessary methods'''

    def __init__(self, params):
        '''Input: params includes data_path, batch_size, mode.
           Return: images [batch_size, height, width, 3],
                 labels [batch_size, num_classes]'''

        self._data_path = params['data_path']
        self._batch_size = params['batch_size']
        self._mode = params['mode']
        self._TrainImageSet = None
        self._TrainLabelSet = None
        self._ValImageSet = None
        self._ValLabelSet = None
        self._TestImageSet = None
        self._TestLabelSet = None
        self._batch_index = 0

        if self._mode == 'Train':
            # Use full training set
            self._total_batches = 50000 / self._batch_size
            # self._total_batches = 45000 / self._batch_size
        else:
            # Use test set
            self._total_batches = 10000 / self._batch_size

        # Split Training set and Validation set in Train Pool
        # Train set 45k, Validation set 5k
        # Don't split if use full training set
        # self._split()
        self._TrainImageSet , self._TrainLabelSet = self._get_TrainPool()
        self._TestImageSet, self._TestLabelSet = self._TestImagePool, self._TestLabelPool = self._get_TestPool()
        self._check_shape()
        self._print_info()

    def _get_data_files(self):
        '''Get all data batch files given data patch, return a list'''

        data_files = []

        search_files = os.path.join(self._data_path, '*')
        data_files = glob.glob(search_files)
        data_files.sort()

        # print('get data files:')
        # print(data_files)

        return data_files

    def _get_data_batch(self, data_file):
        '''Get one data batch(10000 images) filename(path),
           Return(dictionary):
               - data: numpy array(10000x3072), uint8
               - labels: list(10000), range:0-9'''

        try:
            fo = open(data_file, 'rb')
        except IOError as e:
            print('Error: read data batch file %s failed!'%data_file)
        dict = cPickle.load(fo)

        return dict

    def _convert_image_batches(self, batch_dict):
        '''Input: data batch dict from self._get_data_batch,
           Return: images [num_images, height, width, 3],
                 labels [num_images],
           By default, num_images=10000'''

        image_data = batch_dict['data']
        label_data = batch_dict['labels']
        num_images = np.shape(image_data)[0]

        images = np.reshape(image_data, (num_images,32,32,3), order='F')
        images = np.einsum('ijkl->ikjl', images)

        return images, label_data

    def _per_image_standardization(self, image):
        '''Input image: [1, H, W, 3] or [H, W, 3]'''

        # shape check
        if np.ndim(image) == 4 and np.shape(image)[0] != 1:
            sys.exit('Per Image Standardization shape error!')
        else:
            image = np.reshape(image, (32,32,3))
        # Standardization
        image = image.astype(np.int64)
        image_shape = np.shape(image)
        image_mean = np.mean(image)
        num_elements = image_shape[0] * image_shape[1] * image_shape[2]
        variance = np.mean(np.square(image)) - np.square(image_mean)
        variance = np.maximum(variance, 0.0)
        stddev = np.sqrt(variance)
        min_stddev = np.sqrt(num_elements)
        value_scale = np.maximum(min_stddev, stddev)
        value_offset = image_mean

        normed_image = np.subtract(image, value_offset)
        normed_image = np.divide(normed_image, value_scale)

        return normed_image

    def _batch_image_standardization(self, batch_image):
        '''Input batch: [N, H, W, 3]'''

        # shape check
        if np.ndim(batch_image) != 4:
            sys.exit('Batch Image Standardization shape error!')
        # Standardization
        batch_image = batch_image.astype(np.int64)
        batch_shape = np.shape(batch_image)
        num_elements = batch_shape[1] * batch_shape[2] * batch_shape[3]
        image_means = np.mean(batch_image, axis=(1,2,3))
        variances = np.mean(np.square(batch_image), axis=(1,2,3)) - np.square(image_means)
        variances = np.maximum(variances, 0.0)
        stddevs = np.sqrt(variances)
        min_stddevs = np.sqrt(num_elements)
        value_scales = np.maximum(min_stddevs, stddevs)
        value_offsets = image_means

        normed_batch = np.reshape(batch_image, (batch_shape[0], num_elements))
        normed_batch = np.subtract(normed_batch, np.reshape(value_offsets, (batch_shape[0],1)))
        normed_batch = np.divide(normed_batch, np.reshape(value_scales ,(batch_shape[0],1)))
        normed_batch = np.reshape(normed_batch, (batch_shape[0], batch_shape[1], batch_shape[2], batch_shape[3]))

        return normed_batch

    def _get_TrainPool(self):
        '''Generate TrainImagePool and TrainLabelPool'''

        data_files = self._get_data_files()
        image_pool = None
        label_pool = None
        first = True
        for index, each_file in enumerate(data_files):
            if 'data_batch' in each_file:
                data_dict = self._get_data_batch(each_file)
                images_batch, labels_batch = self._convert_image_batches(data_dict)
                if first:
                    image_pool = images_batch
                    label_pool = labels_batch
                    first = False
                else:
                    image_pool = np.concatenate((image_pool, images_batch), axis=0)
                    label_pool = np.concatenate((label_pool, labels_batch), axis=0)

        return image_pool, label_pool

    def _get_TestPool(self):
        '''Generate TestImagePool and TestLabelPool'''

        data_files = self._get_data_files()
        for index, each_file in enumerate(data_files):
            if 'test_batch' in each_file:
                data_dict = self._get_data_batch(each_file)
                image_pool, label_pool = self._convert_image_batches(data_dict)

        return image_pool, label_pool

    def get_val_batch(self):
        '''Return a set of random images from validation set '''

        sample_indices = random.sample(xrange(5000), self._batch_size)

        next_batch_image = np.take(self._ValImageSet, sample_indices, axis=0)
        next_batch_label = np.take(self._ValLabelSet, sample_indices, axis=0)

        return next_batch_image, next_batch_label

    def _split(self):
        '''Split traning pool into TrainSet(45k) and ValSet(5k), randomly pick 5k images for ValSet'''

        TrainImagePool, TrainLabelPool = self._get_TrainPool()
        self._TestImageSet, self._TestLabelSet = self._TestImagePool, self._TestLabelPool = self._get_TestPool()
        sample_indices = random.sample(xrange(50000),5000)

        self._ValImageSet = np.take(TrainImagePool, sample_indices, axis=0)
        self._ValLabelSet = np.take(TrainLabelPool, sample_indices, axis=0)

        self._TrainImageSet = np.delete(TrainImagePool, sample_indices, axis=0)
        self._TrainLabelSet = np.delete(TrainLabelPool, sample_indices, axis=0)

    def shuffle(self):
        '''Shuffle the Train Set 45k images and the corresponding labels'''

        # if use split 45000, 5000
        # sample_indices = np.random.permutation(45000)
        # if use full train set 50000
        sample_indices = np.random.permutation(50000)
        self._TrainImageSet = np.take(self._TrainImageSet, sample_indices, axis=0)
        self._TrainLabelSet = np.take(self._TrainLabelSet, sample_indices, axis=0)

    def flip(self):
        '''Horizontally flip all images'''

        self._TrainImageSet = self._TrainImageSet[:,:,::-1]

    def next_batch(self):

        if self._mode == 'Train':
            next_batch_image, next_batch_label = self._get_next_batch(self._TrainImageSet,self._TrainLabelSet)
        elif self._mode == 'Test':
            next_batch_image, next_batch_label = self._get_next_batch(self._TestImageSet,self._TestLabelSet)
        else:
            sys.exit('load next batch error!')

        return next_batch_image, next_batch_label

    def _get_next_batch(self, image_set, label_set):
        '''Return next batch of Train Set image/label set according to batch_size'''
        # Be careful about the index change at boundaries; or when batch_size if small

        if self._batch_index < self._total_batches:
            next_batch_image = image_set[self._batch_index * self._batch_size: (self._batch_index + 1) * self._batch_size]
            next_batch_label = label_set[self._batch_index * self._batch_size: (self._batch_index + 1) * self._batch_size]
            self._batch_index += 1
        else:
            # if use split
            # full_size = 45000
            if self._mode == 'Train':
                full_size = 50000
            else:
                full_size = 10000
            first_part_image = image_set[self._batch_index * self._batch_size: full_size]
            first_part_label = label_set[self._batch_index * self._batch_size: full_size]
            residul_num = self._batch_size - (full_size - self._batch_index * self._batch_size)
            second_part_image = image_set[0: residul_num]
            second_part_label = label_set[0: residul_num]
            next_batch_image = np.concatenate((first_part_image, second_part_image),axis=0)
            next_batch_label = np.concatenate((first_part_label, second_part_label),axis=0)
            # reset index to 0
            self._batch_index = 0
        next_batch_image = self._batch_image_standardization(next_batch_image)

        return next_batch_image, next_batch_label

    def _get_valset(self):
        '''Get Validation Set image/label set'''

        return self._TestImageSet, self._TestLabelSet

    def _print_info(self):
        '''Print necessary info about the CIFAR10 dataset after initialized'''

        print('Loaded dataset %s:'%self._data_path)
        # print('Training Set {0}, Validation Set {1}, Test Set {2}'.format(np.shape(self._TrainImageSet)[0], np.shape(self._ValImageSet)[0], np.shape(self._TestImageSet)[0]))
        print('Training Set {0}, Test Set {1}'.format(np.shape(self._TrainImageSet)[0], np.shape(self._TestImageSet)[0]))
        print('Shape:')
        print('Training Set: {0}, Label {1}'.format(np.shape(self._TrainImageSet), np.shape(self._TrainLabelSet)))
        # print('Validation Set: {0}, Label {1}'.format(np.shape(self._ValImageSet), np.shape(self._ValLabelSet)))
        print('Test Set: {0}, Label {1}'.format(np.shape(self._TestImageSet), np.shape(self._TestLabelSet)))

    def _check_shape(self):
        '''Do check on data set shapes'''

        flag = False
        try:
            if np.shape(self._TrainImageSet)[0] != np.shape(self._TrainLabelSet)[0]:
                flag = True
            if np.shape(self._TrainImageSet)[0] != 50000:
                flag = True
            # if np.shape(self._ValImageSet)[0] != np.shape(self._ValLabelSet)[0]:
            #    flag = True
            # if np.shape(self._ValImageSet)[0] != 5000:
            #    flag = True
            if np.shape(self._TestImageSet)[0] != np.shape(self._TestLabelSet)[0]:
                flag = True
            if np.shape(self._TestImageSet)[0] != 10000:
                flag = True

            if flag is True:
                raise ValueError('Shape check failed!')
        except ValueError:
            print('Dataset initialization error.')
            raise
