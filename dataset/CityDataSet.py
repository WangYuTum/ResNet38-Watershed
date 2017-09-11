"""
    CityDataSet class:
    Implement loading full size train/val/test RGB images and corresponding GT semantic masks.

    Using batch_size:
        - For loading train images, one can load with batch_size >= 1(default is 4)
        - For loading val/test images, one can only load with batch_size = 1
"""

from __future__ import print_function

from PIL import Image
import os
import sys
import numpy as np
import glob
from collections import namedtuple
from scipy.misc import imsave
from scipy.misc import imread
from scipy.misc import toimage
from scipy import sparse
from scipy.sparse import load_npz

# define a data structure
Label_City = namedtuple( 'Label' , ['name', 'labelId', 'trainId', 'color',] )


class CityDataSet():
    def __init__(self, params):
        '''mode: 'train_sem', 'train_dir', 'val', 'test' '''
        self._mode = params.get('dataset','train_sem')
        self._dir = params.get('data_dir','../data/CityDatabase')
        self._batch_size = params.get('batch_size', 1)

        self._pred_save_path = params.get('pred_save_path','../data/pred_trainIDs')
        self._colored_save_path = params.get('colored_save_path', '../data/pred_colored')
        self._labelIDs_save_path = params.get('labelIDs_save_path', '../data/pred_labelIDs')
        self._num_images = 0
        self._num_batches = 0
        self._batch_idx = 0

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

        # Load dataset indices
        (self._img_indices, self._lbl_indices) = self._load_indicies()

    def _load_indicies(self):

        print('Load %s dataset'%self._mode)
        files_img = []
        files_lbl = []

        # Load train/val/test RGB images
        mode = 'train'
        if self._mode.find('train') != -1:
            mode = 'train'
        else:
            mode = self._mode
        search_img = os.path.join(self._dir,
                                  'leftImg8bit',
                                  mode,'*','*_leftImg8bit.png')
        files_img = glob.glob(search_img)
        files_img.sort()

        # Load GT semantic mask
        if self._mode == 'train_sem':
            search_lbl = os.path.join(self._dir,
                                      'gtFine',
                                      'train',
                                      '*','*_gtFine_labelTrainIds.png')
            files_lbl = glob.glob(search_lbl)
            files_lbl.sort()
        # Load GT watershed direction mask
        if self._mode == 'train_dir':
            search_lbl = os.path.join(self._dir,
                                      'gtFine',
                                      'train',
                                      '*', '*_gtFine_graddir.npz')
            files_lbl = glob.glob(search_lbl)
            files_lbl.sort()

        self._num_images = int(len(files_img))
        self._num_batches = int(self._num_images / self._batch_size)

        print('Loaded images: {0}, GT {1} mask: {2}'.format(len(files_img), self._mode, len(files_lbl)))
        if self._mode == 'train_sem' or self._mode == 'train_dir':
            if len(files_img) != len(files_lbl):
                sys.exit('Number of train images and GTs do not match!')
        return (files_img, files_lbl)

    def next_batch(self):
        """
        Reshape image and label, extend 1st axis for batch dimension
        Return: (image_batch, label_batch)
                image_batch: [batch_size, H, W, 3]
                label_batch: [batch_size, H, W] for semantic label
                             [batch_size, H, W, 2] for graddir label
        """

        batch_idx = self._batch_idx
        total_batches = self._num_batches
        total_images = self._num_images
        image_batch = None
        label_batch = None

        if batch_idx < total_batches:
            image_batch, label_batch = self._get_batch(batch_idx*self._batch_size, (batch_idx+1)*self._batch_size)
            self._batch_idx +=1
        else:
            # At the boundary
            if self._num_images % self._batch_size == 0:
                # restart from beginning
                image_batch, label_batch = self._get_batch(0, self._batch_size)
                self._batch_idx = 1
            else:
                # The first part
                first_image_batch, first_label_batch = _get_batch(batch_idx*self._batch_size, total_images)
                residul_num = self._batch_size - (total_images - batch_idx * self._batch_size)
                second_image_batch, second_label_batch = _get_batch(0, residul_num)
                image_batch = np.concatenate((first_image_batch, second_image_batch))
                if (first_label_batch is not None) and (second_label_batch is not None):
                    label_batch = np.concatenate((first_label_batch, second_label_batch))
                self._batch_idx = 0

        return (image_batch,label_batch)

    def _get_batch(self, start, end):

        for i in range(start, end):
            fname = self._img_indices[i]
            image = self._load_image(fname)
            image = image.reshape(1, *image.shape)
            # Load GT: semantic or graddir
            if self._mode == 'train_sem' or self._mode == 'train_dir':
                gt_fname = self._lbl_indices[i]
                label = self._load_label(gt_fname)
                label = label.reshape(1, *label.shape)
            if i == start:
                image_batch = image
                # Load GT: semantic or graddir
                if self._mode == 'train_sem' or self._mode == 'train_dir':
                    label_batch = label
            else:
                image_batch = np.concatenate((image_batch, image))
                # Load semantic GT
                if self._mode == 'train_sem' or self._mode == 'train_dir':
                    label_batch = np.concatenate((label_batch, label))
        if self._mode.find("train") == -1:
            label_batch = None

        return image_batch, label_batch

    def _load_image(self, fname):
        """
        Load input image and perform standardization:
        Cast to np.float32
        Return: [H, W, 3]
        """
        # print('Loading img:%s'%fname)
        try:
            img = Image.open(fname)
        except IOError as e:
            print('Warning: no image with name %s!!'%fname)

        image = np.array(img, dtype=np.float32)
        ## TODO. Do not switch RGB to BGR since we use pre-trained weights from MXnet.
        ## NOTE: This has effect.
        # image = image[:,:,::-1]     # RGB -> BGR

        ## TODO: per image standardization
        ## NOTE, TODO: This has effect, confirm image preprocessign from MXnext
        # image = self._per_image_standardization(image)

        ## NOTE: use standardization procedure from MXnet implementation
        image = self._transform_image(image)

        return image

    def _load_label(self, fname):
        """
        Return: [H, W] for semantic GT
                [H, W, 2] for graddir GT
        """
        # print('Loading lbl:%s'%fname)
        if self._mode == "train_sem":
            try:
                img = Image.open(fname)
            except IOError as e:
                print('Warning: no image with name %s!!'%fname)
                label = None
                return label
            label = np.array(img, dtype=np.uint8)
        elif self._mode == "train_dir":
            try:
                sparse_grad = load_npz(fname)
            except IOError as e:
                print('Warning: no image with name %s!!'%fname)
                label = None
                return label
            grad = sparse_grad.todense()
            grad = np.array(grad)
            grad = np.reshape(grad, (1024,2048,3))
            # Extract the first 2 channels since the last channel is all-zeros
            label = grad[:,:,:2]
            if np.shape(label) != (1024,2048,2):
                sys.exit("Graddir GT shape error: {0}. Expected shape: {1}".format(np.shape(label), (1024,2048,2)))
        return label

    def _per_image_standardization(self, image):
        '''
            Input image: [1, H, W, 3] or [H, W, 3]
        '''
        # shape check
        if np.ndim(image) == 4 and np.shape(image)[0] != 1:
            sys.exit('Per Image Standardization shape error!')
        else:
            image = np.reshape(image, (1024,2048,3))

        # Standardization
        # use np.int64 in case of numeircal issues
        image = image.astype(np.int64)
        image_shape = np.shape(image)
        image_mean = np.mean(image)
        num_elements = image_shape[0] * image_shape[1] * image_shape[2]
        variance = np.mean(np.square(image)) - np.square(image_mean)
        print("variance: {0}".format(variance))
        print("mean: {0}".format(image_mean))
        variance = np.maximum(variance, 0.0)
        stddev = np.sqrt(variance)
        min_stddev = np.sqrt(num_elements)
        value_scale = np.maximum(min_stddev, stddev)
        value_offset = image_mean

        normed_image = np.subtract(image, value_offset)
        normed_image = np.divide(normed_image, value_scale)
        normed_image = normed_image.astype(np.float32)

        return normed_image

    def _transform_image(self, image):
        '''
            The standardization procedure used in MXnet implementation.
        '''
        # color scale
        image *= (1.0 / 255)

        # subtract mean for each channel
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        image -= mean

        # divide std for each channel
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        image /= std

        return image

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
        pred_in shape: [1, H, W] -> need to reshape to [H, W] to save .png

        Note that this function only works for inference: read one image at a time.
        '''
        # Since self._batch_idx is already increased by 1, need to decrease 1.
        img_idx = self._batch_idx - 1
        img_inx = self._img_indices[img_idx].split('/')
        # ../data/CityDatabase/gtFine/{val, test}/frankfurt/fname.png
        fname = img_inx[6]
        fname = fname.split('_')
        fname = fname[0]+'_'+fname[1]+'_'+fname[2]+'_trainIds.png'
        save_path = os.path.join(self._pred_save_path,fname)

        # Reshape to [H,W]
        pred_in = np.reshape(pred_in, (pred_in.shape[1], pred_in.shape[2]))
        # Save .png, don't rescale
        toimage(pred_in, high=18, low=0, cmin=0, cmax=18).save(save_path)
        print("TrainIDs prediction saved to %s "%save_path)


    def pred_to_labelID(self):
        '''
        For evaluation purpose:
        convert prediction (*_trainIDs.png) to
        evaluation format (*_labelID.png).

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


# Test example
'''
data_config = {'city_dir':"./data/CityDatabase",
                     'randomize': True,
                     'seed': None,
                     'dataset': 'test'}
dt = CityDataSet(data_config)
(img,lbl)=dt.next_batch()
print(img.shape,' ',lbl==None)
'''
