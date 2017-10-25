'''
    The data pipeline for cityscape encapsulated as a class object.
    This code is reconstructed and deprecated CityDataSet.py

    NOTE that the data pipeline is currently only usable for semantic training/validation.
    TODO:
        * Pipeline for training graddir
        * Pipeline for training discretized watershed transform
'''

#NOTE: The .tfrecord file is located on /work/wangyu/ on hpccremers4.
#NOTE: The original images/gts are located on ~/ResNet38-Watershed/data/CityDatabase

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import sys
import os
import glob
from scipy.misc import imsave
from scipy.misc import toimage
from scipy.misc import imread
from collections import namedtuple

# Label mapping of cityscape dataset
Label_City = namedtuple( 'Label' , ['name', 'labelId', 'trainId', 'color',] )

class CityDataSet():
    def __init__(self, params):
        '''mode: 'train_sem', 'val_sem','test_sem', 'train_grad', 'val_grad' '''

        self._mode = params.get('mode','train_sem')
        self._batch_size = params.get('batch_size', 4)
        self._train_size = 2975
        self._TFrecord_file = None
        self._dataset = None
        self._valimg_indices = None
        self._batch_idx = 0

        self._pred_save_path = params.get('pred_save_path','../data/pred_trainIDs')
        self._colored_save_path = params.get('colored_save_path', '../data/pred_colored')
        self._labelIDs_save_path = params.get('labelIDs_save_path', '../data/pred_labelIDs')

        # Create mapping of (lable_name, id, color)
        self._labels = [
            Label_City(  'road'          ,   7,  0, (128, 64,128) ),
            Label_City(  'sidewalk'      ,   8,  1, (244, 35,232) ),
            Label_City(  'building'      ,   11,  2, ( 70, 70, 70) ),
            Label_City(  'wall'          ,   12,  3, (102,102,156) ),
            Label_City(  'fence'         ,   13,  4, (190,153,153) ),
            Label_City(  'pole'          ,   17,  5, (153,153,153) ),
            Label_City(  'traffic light' ,   19,  6, (250,170, 30) ),
            Label_City(  'traffic sign'  ,   20,  7, (220,220,  0) ),
            Label_City(  'vegetation'    ,   21,  8, (107,142, 35) ),
            Label_City(  'terrain'       ,   22,  9, (152,251,152) ),
            Label_City(  'sky'           ,   23, 10, ( 70,130,180) ),
            Label_City(  'person'        ,   24, 11, (220, 20, 60) ),
            Label_City(  'rider'         ,   25, 12, (255,  0,  0) ),
            Label_City(  'car'           ,   26, 13, (  0,  0,142) ),
            Label_City(  'truck'         ,   27, 14, (  0,  0, 70) ),
            Label_City(  'bus'           ,   28, 15, (  0, 60,100) ),
            Label_City(  'train'         ,   31, 16, (  0, 80,100) ),
            Label_City(  'motorcycle'    ,   32, 17, (  0,  0,230) ),
            Label_City(  'bicycle'       ,   33, 18, (119, 11, 32) ),
            Label_City(  'void'          ,   19, 19, (  0,  0,  0) )
        ]
        self._trainId2Color = [label.color for label in self._labels]
        self._trainId2labelId = [label.labelId for label in self._labels]

        # Build up the pipeline
        if self._mode == 'train_sem':
            self._TFrecord_file = '/work/wangyu/cityscape_train.tfrecord'
        elif self._mode == 'val_sem':
            self._TFrecord_file = '/work/wangyu/cityscape_val.tfrecord'
            self._valimg_indices = self._load_valimg_indicies()
        # TODO
        elif self._mode == 'test_sem':
            self._TFrecord_file = '/work/wangyu/cityscape_test.tfrecord'
        elif self._mode == 'train_grad':
            self._TFrecord_file = '/work/wangyu/cityscape_train.tfrecord'
        elif self._mode == 'val_grad':
            self._TFrecord_file = '/work/wangyu/cityscape_val.tfrecord'
        else:
            sys.exit('No valid mode!')
        self._dataset = self._build_pipeline()
        if self._dataset is None:
            sys.exit('Dataset pipeline is None!')

    def _parse_single_record(self, record):
        '''Given a record (tf.string, serilized) , and parse_features.
            Return: data_dict'''

        # TFrecords format for cityscape dataset
        record_features = {
            "height": tf.FixedLenFeature([1],tf.int64),
            "width": tf.FixedLenFeature([1],tf.int64),
            "img": tf.FixedLenFeature([1024, 2048, 3],tf.int64),
            "sem_gt": tf.FixedLenFeature([1024, 2048, 1],tf.int64),
            "grad_gt": tf.FixedLenFeature([1024, 2048, 2],tf.float32),
            "wt_gt": tf.FixedLenFeature([1024, 2048, 1],tf.int64),
        }

        data_dict = {}
        out_data = tf.parse_single_example(serialized=record, features=record_features)

        # Cast RGB pixels to tf.float32
        data_dict['img'] = tf.cast(out_data['img'], tf.float32)
        data_dict['sem_gt'] = tf.cast(out_data['sem_gt'], tf.int32)
        data_dict['grad_gt'] = tf.cast(out_data['grad_gt'], tf.float32)
        # data_dict['wt_gt'] = tf.cast(out_data['wt_gt'], tf.int32)

        return data_dict

    def _image_standardization(self, example):
        '''Given a parsed example: dictionary
            Return: a dictionary, where RGB image is standardized.

        NOTE: This is the standardization procedure used in MXnet implementation.
        '''

        # color scale
        rgb_img = example['img']
        rgb_img *= (1.0 / 255)

        # subtract mean for each channel
        mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        rgb_img -= mean

        # divide std for each channel
        std = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
        rgb_img /= std

        # Pack the result
        transformed = {}
        transformed['img'] = rgb_img
        transformed['sem_gt'] = example['sem_gt']
        transformed['grad_gt'] = example['grad_gt']
        # transformed['wt_gt'] = example['wt_gt']

        return transformed

    def _sem_train_transform(self, example):
        '''Given a standardized example: dictonary
            Return: a dictionary, where RGB_image and Sem_gt is transformed

            Transformation:
                * Randomly resize image/sem_gt in range: [0.7, 1.3]
                * Randomly crop image/sem_gt to: [504, 504]
                * Randomly flip image/sem_gt together
            After this transformation:
                * All (image, label) has the same shape: ([504,504,3], [504,504,1])
        '''
        # Randomly resize
        #TODO: resize method and rand ratio need to be investiaged/compared to original paper
        [ratio] = np.random.randint(7,13,1)
        new_H = tf.cast(1024*ratio/10, tf.int32)
        new_W = tf.cast(2048*ratio/10, tf.int32)
        new_size = [new_H, new_W]
        label = tf.image.resize_images(example['sem_gt'], new_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.resize_images(example['img'], new_size, tf.image.ResizeMethod.BILINEAR)
        label = tf.cast(label, tf.float32) #NOTE, image is already normalized, cannot be converted to tf.int32
        stacked = tf.concat([image, label], axis=-1) #NOTE, shape [H, W, 4]

        # Randomly crop and flip
        stacked = tf.random_crop(stacked, [504, 504, 4])
        stacked = tf.image.random_flip_left_right(stacked)

        # Pack the result
        image = stacked[:,:,0:3]
        label = tf.cast(stacked[:,:,3:4], tf.int32)
        transformed = {}
        transformed['img'] = image
        transformed['sem_gt'] = label
        # transformed['grad_gt'] = example['grad_gt']
        # transformed['wt_gt'] = example['wt_gt']

        return transformed

    def _grad_train_transform(self, example):
        '''Given a standardized example: dictonary
            Return: a dictionary, where RGB_image and sem_gt/grad_gt is transformed

            Transformation:
                * Randomly flip image/sem_gt/grad_gt together
                * Resize image to [512, 1024] fit TitanX 12GB memory
                * Resize sem_gt/grad_gt by 1/(2*8) for loss calculation, [64,128]
            After this transformation:
                * Image has shape: [512,1024,3], tf.float32
                * sem_gt has shape: [64,128,1], tf.int32
                * grad_gt has shape: [64,128,2], tf.float32
        '''
        # Randomly flip
        sem_gt = tf.cast(example['sem_gt'], tf.float32)
        stacked = tf.concat([example['img'], sem_gt, example['grad_gt']], axis=-1) #NOTE, shape [H, W, 6]
        stacked = tf.image.random_flip_left_right(stacked)

        image = tf.image.resize_images(stacked[:,:,0:3], [512,1024], tf.image.ResizeMethod.BILINEAR)
        sem_gt = tf.image.resize_images(stacked[:,:,3:4], [64,128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        grad_gt = tf.image.resize_images(stacked[:,:,4:6], [64,128], tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        # Pack the result
        sem_gt = tf.cast(sem_gt, tf.int32)
        transformed = {}
        transformed['img'] = image
        transformed['sem_gt'] = sem_gt
        transformed['grad_gt'] = grad_gt
        # transformed['wt_gt'] = example['wt_gt']

        return transformed

    def _build_semtrain_pipeline(self, TFrecord_file):
        '''
            Given the .tfrecord path, build a datapipeline using member functions
            Return: A TF dataset object

            NOTE: The .tfrecord file is compressed using "GZIP"
        '''
        dataset = tf.contrib.data.TFRecordDataset(TFrecord_file, "GZIP")
        dataset = dataset.repeat()
        dataset = dataset.map(self._parse_single_record, num_threads=3, output_buffer_size=9)
        dataset = dataset.map(self._image_standardization, num_threads=3, output_buffer_size=9)
        dataset = dataset.map(self._sem_train_transform, num_threads=3, output_buffer_size=9)
        dataset = dataset.shuffle(buffer_size=1500)
        dataset = dataset.batch(self._batch_size)

        return dataset

    def _build_semval_pipeline(self, TFrecord_file):
        '''
            Given the .tfrecord path, build a datapipeline using member functions
            Return: A TF dataset object

            NOTE: The .tfrecord file is compressed using "GZIP"
        '''

        #NOTE: Only go through .tfrecord once and no shuffle
        dataset = tf.contrib.data.TFRecordDataset(TFrecord_file, "GZIP")
        dataset = dataset.map(self._parse_single_record, num_threads=4, output_buffer_size=8)
        dataset = dataset.map(self._image_standardization, num_threads=4, output_buffer_size=8)
        dataset = dataset.batch(self._batch_size)

        return dataset

    def _build_gradtrain_pipeline(self, TFrecord_file):

        '''
            Given the .tfrecord path, build a datapipeline using member functions
            Return: A TF dataset object

            NOTE: The .tfrecord file is compressed using "GZIP"
        '''
        dataset = tf.contrib.data.TFRecordDataset(TFrecord_file, "GZIP")
        dataset = dataset.repeat()
        dataset = dataset.map(self._parse_single_record, num_threads=2, output_buffer_size=8)
        dataset = dataset.map(self._image_standardization, num_threads=2, output_buffer_size=8)
        dataset = dataset.map(self._grad_train_transform, num_threads=2, output_buffer_size=8)
        dataset = dataset.shuffle(buffer_size=1500)
        dataset = dataset.batch(self._batch_size)

        return dataset

    def _build_gradval_pipeline(self, TFrecord_file):

        '''
            Given the .tfrecord path, build a datapipeline using member functions
            Return: A TF dataset object

            NOTE: The .tfrecord file is compressed using "GZIP"
        '''

        #NOTE: Only go through .tfrecord once and no shuffle
        dataset = tf.contrib.data.TFRecordDataset(TFrecord_file, "GZIP")
        dataset = dataset.map(self._parse_single_record, num_threads=4, output_buffer_size=8)
        dataset = dataset.map(self._image_standardization, num_threads=4, output_buffer_size=8)
        dataset = dataset.batch(self._batch_size)

        return dataset

    def _build_pipeline(self):

        if self._mode == 'train_sem':
            dataset = self._build_semtrain_pipeline(TFrecord_file=self._TFrecord_file)
            print('Train_sem pipeline built. Load tfrecord: {}'.format(self._TFrecord_file))
        elif self._mode == 'val_sem':
            dataset = self._build_semval_pipeline(TFrecord_file=self._TFrecord_file)
        elif self._mode == 'train_grad':
            dataset = self._build_gradtrain_pipeline(TFrecord_file=self._TFrecord_file)
        elif self._mode == 'val_grad':
            dataset = self._build_gradval_pipeline(TFrecord_file=self._TFrecord_file)
        else:
            sys.exit('Mode {} is not supported.'.format(self._mode))

        return dataset

    def next_batch(self):
        '''
            Given a TF dataset object.
            Return: an operator which retrieves the next batch

            Format of next_batch (a python dictionary):
                * next_batch['img'] = [batch_size, H, W, 3], tf.float32
                * next_batch['sem_gt'] = [batch_size, H, W, 1], tf.int32
                * next_batch['grad_gt'] = [batch_size, H, W, 2], tf.float32
                * next_batch['wt_gt'] = [batch_size, H, W, 1], tf.int32
        '''
        batch_iterator = self._dataset.make_one_shot_iterator()
        next_batch = batch_iterator.get_next()

        return next_batch

    ####################################################################
    ## The following functions are used to transform/visualize the predictions

    def _padding_func(self, vector, iaxis_pad_width, iaxis, kwargs):
        '''
            Used by pred_to_color
        '''
        if iaxis == 3:
            idx = vector[0]
            values = self._trainId2Color[idx]
            vector[-iaxis_pad_width[1]:] = values

        return vector

    def pred_to_color(self):
        '''
            Input:  self._pred_save_path, original prediction images with trainIDs. Each image has shape [H,W]
            Output: self._colored_save_path, converted color prediction images. Each image need to be [H,W,3]
        '''
        search_img = os.path.join(self._pred_save_path, '*.png')
        img_files = glob.glob(search_img)
        img_files.sort()

        for i in range(len(img_files)):
            fname = img_files[i]
            pred_in = imread(fname)
            # Pad with RGB channels, producing [1, Height, Width, 4]
            pred_in = pred_in[np.newaxis, ..., np.newaxis]
            pred = np.lib.pad(pred_in, ((0,0),(0,0),(0,0),(0,3)), self._padding_func)
            # Slice RGB channels
            pred = pred[:,:,:,1:4]
            H = pred.shape[1]
            W = pred.shape[2]
            pred = np.reshape(pred, (H,W,3) )

            # write to .png file
            img_inx = fname.split('/')
            # ../data/pred_trainIDs/fname.png
            img_inx = img_inx[3]
            img_inx = img_inx.replace('trainIDs', 'colored')
            save_color_path = self._colored_save_path + '/' + img_inx
            imsave(save_color_path, pred)
            print('Colored prediction saved to %s '%save_color_path)

        return None

    def save_trainID_img(self, pred_in):
        '''
            This method is meant to save original prediction into .png
            pred_in shape: [batch_size, H, W] -> need to reshape to [H, W] to save .png

            Note that this function only works during inference.
        '''
        for i in range(self._batch_idx, self._batch_idx+self._batch_size):
            img_inx = self._valimg_indices[i].split('/')
            # eg, ../data/CityDatabase/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
            fname = img_inx[6]
            fname = fname.split('_')
            fname = fname[0]+'_'+fname[1]+'_'+fname[2]+'_trainIds.png'
            save_path = os.path.join(self._pred_save_path,fname)

            # Reshape to [H,W]
            pred_label = np.reshape(pred_in[i%self._batch_size,:,:], (1024,2048))
            # Save .png, don't rescale
            toimage(pred_label, high=18, low=0, cmin=0, cmax=18).save(save_path)
            print("TrainIDs prediction saved to %s "%save_path)

        # Update batch index
        self._batch_idx += self._batch_size


    def pred_to_labelID(self):
        '''
            For evaluation purpose:
              * Converting prediction (*_trainIDs.png) to evaluation format (*_labelID.png).

            Input:  self._pred_save_path, original prediction images. Each image has shape [H,W]
            Output: self._labelIDs_save_path, converted labelled images. Each image need to be [H,W]
        '''

        search_path = os.path.join(self._pred_save_path, '*')
        files_img = glob.glob(search_path)
        files_img.sort()

        print("TrainIDs prediction has %d images."%len(files_img))
        for idx in range(len(files_img)):
            img = imread(files_img[idx])
            H = img.shape[0]
            W = img.shape[1]
            image = np.array(img, dtype=np.uint8)
            image = np.reshape(image, (H*W))

            for i in range(H*W):
                image[i] = self._trainId2labelId[image[i]]

            # Restore to original image size
            image = np.reshape(image, (H, W))
            output_img = files_img[idx].replace(self._pred_save_path, self._labelIDs_save_path)
            output_img = output_img.replace('trainIds', 'labelIds')

            imsave(output_img, image)
            print("LabelIDs prediction saved to %s"%output_img)

    def _load_valimg_indicies(self):
        '''
            Load val image names for inference.
        '''
        print('Loading val set indices...')
        files_img = []

        if self._mode != 'val_sem':
            sys.exit('Error: Wrong mode during validation.')

        search_img = os.path.join('../data/CityDatabase', 'leftImg8bit', 'val',
                                  '*', '*_leftImg8bit.png')
        files_img = glob.glob(search_img)
        files_img.sort()

        print('Loaded val image indices: {}'.format(len(files_img)))
        return (files_img)

# Only used for test purpose
#def main():
#
#    TFrecord_file = "/work/wangyu/cityscape_train.tfrecord"
#    train_dataset = build_pipeline(TFrecord_file=TFrecord_file)
#    next_batch_op = next_batch(train_dataset)
#
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        next_batch_ = sess.run(next_batch_op)
#        print(next_batch_['sem_gt'].shape)
#
#    # Visualize the batch data
#    sv_img0 = next_batch_['img'][0,:,:,:]
#    imsave('recons_img_tfrecords_0.png', sv_img0)
#    sv_sem0 = next_batch_['sem_gt'][0,:,:,0]
#    imsave('recons_sem_tfrecords_0.png', sv_sem0)
#
#    sv_img1 = next_batch_['img'][1,:,:,:]
#    imsave('recons_img_tfrecords_1.png', sv_img1)
#    sv_sem1 = next_batch_['sem_gt'][1,:,:,0]
#    imsave('recons_sem_tfrecords_1.png', sv_sem1)
#
#if __name__ == "__main__":
#    main()
