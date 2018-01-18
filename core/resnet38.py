#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import nn
import sys
import os
sys.path.append("..")
import data_utils as dt

class ResNet38:
    def __init__(self, params):

        ## use pre-trained A1 model on Cityscapes unless specified
        weight_path = params.get('feed_weight', '../data/trained_weights/pretrained_ResNet38a1_city.npy')
        self._data_format = params.get('data_format', "NCHW")
        self._weight_dict = dt.load_weight(weight_path)
        self._var_dict = {}

        self._num_classes = params.get('num_classes', 19)

    def _build_model(self, image, sem_gt, is_train=False):
        '''If is_train, save weight to self._var_dict,
           otherwise, don't save weights
           '''
        model = {}
        feed_dict = self._weight_dict
        if is_train:
            var_dict = self._var_dict
        else:
            var_dict = None

        if is_train:
            dropout = True
        else:
            dropout = False

        if self._data_format == "NCHW":
            image = tf.transpose(image, [0, 3, 1, 2])
            sem_gt = tf.transpose(sem_gt, [0, 3, 1, 2])

        shape_dict = {}
        shape_dict['B0'] = [3,3,3,64]

        # The sharable conv features: 36 conv layers
        with tf.variable_scope('shared'):
            # B0: [H,W,3] -> [H,W,64]
            with tf.variable_scope('B0'):
                model['B0'] = nn.conv_layer(self._data_format, image, feed_dict, 1, 'SAME',
                                            shape_dict['B0'], var_dict)

            # B2_1: [H,W,64] -> [H/2, W/2, 128]
            shape_dict['B2'] = {}
            shape_dict['B2']['side'] = [1,1,64,128]
            shape_dict['B2']['convs'] = [[3,3,64,128],[3,3,128,128]]
            with tf.variable_scope('B2_1'):
                model['B2_1'] = nn.ResUnit_downsample_2convs(self._data_format, model['B0'],
                                                             feed_dict,
                                                             shape_dict['B2'],
                                                             var_dict=var_dict)
            # B2_2, B2_3: [H/2, W/2, 128]
            for i in range(2):
                with tf.variable_scope('B2_'+str(i+2)):
                    model['B2_'+str(i+2)] = nn.ResUnit_2convs(self._data_format, model['B2_'+str(i+1)], feed_dict,
                                                              shape_dict['B2']['convs'][1],
                                                              var_dict=var_dict)

            # B3_1: [H/2, W/2, 128] -> [H/4, W/4, 256]
            shape_dict['B3'] = {}
            shape_dict['B3']['side'] = [1,1,128,256]
            shape_dict['B3']['convs'] = [[3,3,128,256],[3,3,256,256]]
            with tf.variable_scope('B3_1'):
                model['B3_1'] = nn.ResUnit_downsample_2convs(self._data_format, model['B2_3'],
                                                             feed_dict,
                                                             shape_dict['B3'],
                                                             var_dict=var_dict)
            # B3_2, B3_3: [H/4, W/4, 256]
            for i in range(2):
                with tf.variable_scope('B3_'+str(i+2)):
                    model['B3_'+str(i+2)] = nn.ResUnit_2convs(self._data_format, model['B3_'+str(i+1)], feed_dict,
                                                              shape_dict['B3']['convs'][1],
                                                              var_dict=var_dict)
            # B4_1: [H/4, W/4, 256] -> [H/8, W/8, 512]
            shape_dict['B4'] = {}
            shape_dict['B4']['side'] = [1,1,256,512]
            shape_dict['B4']['convs'] = [[3,3,256,512],[3,3,512,512]]
            with tf.variable_scope('B4_1'):
                model['B4_1'] = nn.ResUnit_downsample_2convs(self._data_format, model['B3_3'],
                                                                 feed_dict,
                                                                 shape_dict['B4'],
                                                                 var_dict=var_dict)
            # B4_2 ~ B4_6: [H/8, W/8, 512]
            for i in range(5):
                with tf.variable_scope('B4_'+str(i+2)):
                    model['B4_'+str(i+2)] = nn.ResUnit_2convs(self._data_format, model['B4_'+str(i+1)],
                                                                   feed_dict,
                                                                   shape_dict['B4']['convs'][1],
                                                                   var_dict=var_dict)
#        with tf.variable_scope('semantic'):
#            # B5_1: [H/8, W/8, 512] -> [H/8, W/8, 1024]
#            shape_dict['B5_1'] = {}
#            shape_dict['B5_1']['side'] = [1,1,512,1024]
#            shape_dict['B5_1']['convs'] = [[3,3,512,512],[3,3,512,1024]]
#            with tf.variable_scope('B5_1'):
#                model['B5_1'] = nn.ResUnit_hybrid_dilate_2conv(self._data_format, model['B4_6'],
#                                                                   feed_dict,
#                                                                   shape_dict['B5_1'],
#                                                                   var_dict=var_dict)
#            # B5_2, B5_3: [H/8, W/8, 1024]
#            # Shape for B5_2, B5_3
#            shape_dict['B5_2_3'] = [[3,3,1024,512],[3,3,512,1024]]
#            for i in range(2):
#                with tf.variable_scope('B5_'+str(i+2)):
#                    model['B5_'+str(i+2)] = nn.ResUnit_full_dilate_2convs(self._data_format, model['B5_'+str(i+1)],
#                                                      feed_dict, shape_dict['B5_2_3'],
#                                                      var_dict=var_dict)
#
#            # B6: [H/8, W/8, 1024] -> [H/8, W/8, 2048]
#            shape_dict['B6'] = [[1,1,1024,512],[3,3,512,1024],[1,1,1024,2048]]
#            with tf.variable_scope('B6'):
#                model['B6'] = nn.ResUnit_hybrid_dilate_3conv(self._data_format, model['B5_3'],
#                                                                 feed_dict,
#                                                                 shape_dict['B6'],
#                                                                 dropout=dropout,
#                                                                 var_dict=var_dict)
#            # B7: [H/8, W/8, 2048] -> [H/8, W/8, 4096]
#            shape_dict['B7'] = [[1,1,2048,1024],[3,3,1024,2048],[1,1,2048,4096]]
#            with tf.variable_scope('B7'):
#                model['B7'] = nn.ResUnit_hybrid_dilate_3conv(self._data_format, model['B6'],
#                                                                 feed_dict,
#                                                                 shape_dict['B7'],
#                                                                 dropout=dropout,
#                                                                 var_dict=var_dict)
#            # Semantic classification layer
#            shape_dict['semantic'] = [[3,3,4096,512],[3,3,512,self._num_classes]]
#            model['semantic'] = nn.ResUnit_tail(self._data_format, model['B7'], feed_dict,
#                                                shape_dict['semantic'], var_dict)
#
        # Gating op, semantic comes from semantic layer
        # sem_out = tf.expand_dims(tf.argmax(tf.nn.softmax(model['semantic'], dim=1), axis=1), axis=1)
        # if self._data_format == 'NCHW':
        #     pred_sem = tf.transpose(sem_out, [0, 2, 3, 1]) #NOTE, pred_sem is [batch_size, 64, 128, 1]
        # pred_sem_sum = tf.summary.image('pred_sem', tf.cast(pred_sem, tf.float16))
        # Gating uses sem_gt: [batch, C, H, W]
        model['gated_feat'] = self._gate(sem_gt, model['B4_6']) # sem_out [batch, 1, 64, 128], feature: [batch, 512, 64, 128]

        with tf.variable_scope("graddir"):
            # Further feature extractors
            shape_dict['grad_convs1'] = [[3,3,512,512],[3,3,512,512]]
            with tf.variable_scope('convs1'):
                model['grad_convs1'] = nn.grad_convs(self._data_format, model['gated_feat'], feed_dict,
                                                     shape_dict['grad_convs1'], var_dict)
            # The normalize the output as the magnitued of GT
            shape_dict['grad_convs2'] = [[1,1,512,256],[1,1,256,256],[1,1,256,2]]
            with tf.variable_scope('convs2'):
                model['grad_convs2'] = nn.grad_norm(self._data_format, model['grad_convs1'], feed_dict,
                                                    shape_dict['grad_convs2'], var_dict)
            # Normalize the output to have unit vectors
            model['grad_norm'] = nn.norm(self._data_format, model['grad_convs2'])

        return model

    def _gate(self, sem_input, feat_input):
        ''' This function takes inputs as semantic result and feature maps,
            returns gated feature maps, where non-relevant classes on the
            feature maps are set to zero
            Input: sem_input [batch_size, 1, H, W]
                   feat_input [batch_size, C, H, W]
            Output: gated feature maps [batch_size, H, W, C]
            NOTE: The default data format is "NCHW", the function also works for "NHWC" without any changes in code.
        '''

        # NOTE: Gate all classes: [person, rider, car, truck, bus, train, motorcycle, bicycle]
        # [11, 12, 13, 14, 15, 16, 17, 18]
        bool_mask0 = tf.equal(sem_input, 11) # shape [batch_size, 1, H, W]
        bool_mask1 = tf.equal(sem_input, 12) # shape [batch_size, 1, H, W]
        bool_stack = tf.stack([bool_mask0, bool_mask1],axis=1) # shape [batch_size, 1,2, H, W]
        for class_num in range(6):
            new_bool = tf.expand_dims(tf.equal(sem_input, class_num+13), axis=1)
            bool_stack = tf.concat([bool_stack, new_bool], axis=1) # shape: [batch_size, 1,2, H, W]
        bool_mask = tf.reduce_any(bool_stack, axis=1) # shape: [batch_size, 1, H, W]
        sem_bin = tf.cast(bool_mask, tf.float32) #NOTE [batch_size, 1, H, W], zeros/ones

        ## Gate
        gated_feat = tf.multiply(feat_input, sem_bin) #NOTE [batch_size, C, H, W]

        return gated_feat


    def _upsample(self, input_tensor, new_size):
        ''' Upsampling using Bilinear interpolation
            Input: A tensor [batch_size, H, W, C]
                   new_size: python/numpy array [new_H, new_W]
            Return: upsampled tensor
        '''
        with tf.variable_scope('Bilinear'):
            upsampled = nn.bilinear_upscore_layer(input_tensor, new_size)

        return upsampled

    def _weight_decay(self, decay_rate):
        '''Compute weight decay loss for convolution kernel and fully connected
        weigths, excluding trainable variables of BN layer'''

        l2_losses = []
        for var in tf.trainable_variables():
            if var.op.name.find('kernel') or var.op.name.find('bias'):
                l2_losses.append(tf.nn.l2_loss(var))

        return tf.multiply(decay_rate, tf.add_n(l2_losses))

    def num_parameters(self):
        '''Compute the number of trainable parameters. Note that it MUST be called after the graph is built'''

        return np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])

        # flipped_img = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped_img)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
             # default learning rate for Adam: 0.001
             # train_op = tf.train.AdamOptimizer().minimize(total_loss)

    def train_grad(self, image, sem_gt, grad_gt, params):
        ''' This function only trains graddir branch.
            Input: Image [batch_size, 512, 1024, 3]
                   sem_gt [batch_size, 64, 128, 1]
                   grad_gt [batch_size, 64, 128, 3]
                        * [batch_size, 64, 128, 0:2] is the grad_gt
                        * [batch_size, 64, 128, 2:3] is the inverse of sqrt(area)
                   params: decay_rate, lr
        '''

        model = self._build_model(image, sem_gt, is_train=True)

        ## NOTE Get semantic loss
        # sem_loss = self._get_sem_loss(model, sem_gt, params)

        pred = model['grad_norm'] #NOTE pred  [batch_size, 2, 64,128] if "NCHW"
        if self._data_format == 'NCHW':
            pred = tf.transpose(pred, [0, 2, 3, 1]) #NOTE, pred is [batch_size, 64, 128, 2]

        ## The predicted graddir and GT are already normalized
        product = tf.reduce_sum(tf.multiply(pred,grad_gt[:,:,:,0:2]), axis=3) #NOTE product [batch_size, 64,128]
        product = tf.maximum(product, -0.99)
        product = tf.minimum(product, 0.99)
        cos_out = tf.acos(product)
        sem_gt = tf.reshape(sem_gt, [params['batch_size'],64,128]) #NOTE sem_gt [batch_size,64,128]
        # Get valid pixels to compute: [person, rider, car, truck, bus, train, motorcycle, bicycle]
        # [11, 12, 13, 14, 15, 16, 17, 18]
        bool_mask0 = tf.equal(sem_gt, 11) # shape [batch_size, 64, 128]
        bool_mask1 = tf.equal(sem_gt, 12) # shape [batch_size, 64, 128]
        bool_stack = tf.stack([bool_mask0, bool_mask1],axis=-1) # shape [batch_size, 64, 128, 2]
        for class_num in range(6):
            new_bool = tf.expand_dims(tf.equal(sem_gt, class_num+13), axis=-1) # shape [batch_size, 64, 128, 1]
            bool_stack = tf.concat([bool_stack, new_bool], axis=-1) # shape: [batch_size, 64, 128, 2+(class_num+1)]
        bool_mask = tf.reduce_any(bool_stack, axis=-1) # shape: [batch_size, 64, 128]
        valid_grad_weight = tf.boolean_mask(grad_gt[:,:,:,2:3], bool_mask)

        # if there's no valid label, set loss to 0.0
        valid_cos_out = tf.cond(tf.equal(tf.reduce_sum(tf.cast(bool_mask, tf.int32)), 0), lambda: 0.0, lambda: tf.boolean_mask(cos_out, bool_mask))
        loss_grad = tf.reduce_mean(tf.square(valid_cos_out) * valid_grad_weight * 100.0)
        loss_grad_valid = tf.cond(tf.is_nan(loss_grad), lambda: 0.0, lambda: loss_grad)
        loss_grad_total = loss_grad_valid + self._weight_decay(params['decay_rate'])

        ## NOTE Total loss
        # loss_total = sem_loss + loss_grad_total
        loss_total = loss_grad_total

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(params['lr']).minimize(loss_total)
            # train_step = tf.train.MomentumOptimizer(params['lr'],0.9).minimize(loss_total)

        ###
        pred_grad_sum = tf.summary.image('grad_out', tf.concat([pred, tf.zeros([params['batch_size'], 64, 128, 1])], axis=-1))
        ###

        return train_step, loss_total

    def _get_sem_loss(self, model, sem_gt, params):
        '''
            Get the semantic loss.
            Input: model
                   sem_gt: [batch_size, 64, 128, 1]
                   params
        '''
        pred = model['semantic'] # shape [batch_size, 19, 64, 128]
        new_shape = [params['batch_size'], self._num_classes, 64*128]
        pred = tf.reshape(pred, new_shape) # [batch_size, 19, 64*128]
        if self._data_format == 'NCHW':
            pred = tf.transpose(pred, [0, 2, 1]) # [batch_size, 64*128,19]
        sem_gt = tf.reshape(sem_gt, [params['batch_size'] ,64*128]) # [batch_size, 64*128]
        void_bool = tf.equal(sem_gt, 19)
        valid_bool = tf.logical_not(void_bool)
        valid_label = tf.boolean_mask(sem_gt, valid_bool)
        valid_pred = tf.boolean_mask(pred, valid_bool)

        loss_sem = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_label, logits=valid_pred))

        return loss_sem

    def inf(self, image, sem_gt):
        ''' Input: Image [batch_size, 1024, 2048, 3]
                   sem_gt [batch_size, 1024, 2048, 1]
            Output: upsampled graddir result [batch_size, 1024, 2048, 2]
        '''
        small_size = [128, 256]
        sem_gt0 = tf.image.resize_images(sem_gt, small_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        model = self._build_model(image, sem_gt0, is_train=False)

        pred = model['grad_norm'] #NOTE, pred is [batch_size, 2, 128, 256] if "NCHW"
        if self._data_format == 'NCHW':
            pred = tf.transpose(pred, [0, 2, 3, 1]) #NOTE, pred is [batch_size, 128, 256, 2]
        pred = self._upsample(pred, [1024,2048]) # [batch_size, 1024, 2048, 2]

        # gate
        pred = self._gate(sem_gt, pred)

        return pred


