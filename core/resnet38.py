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
            # B5_1: [H/8, W/8, 512] -> [H/8, W/8, 1024]
            shape_dict['B5_1'] = {}
            shape_dict['B5_1']['side'] = [1,1,512,1024]
            shape_dict['B5_1']['convs'] = [[3,3,512,512],[3,3,512,1024]]
            with tf.variable_scope('B5_1'):
                model['B5_1'] = nn.ResUnit_hybrid_dilate_2conv(self._data_format, model['B4_6'],
                                                                   feed_dict,
                                                                   shape_dict['B5_1'],
                                                                   var_dict=var_dict)
            # B5_2, B5_3: [H/8, W/8, 1024]
            # Shape for B5_2, B5_3
            shape_dict['B5_2_3'] = [[3,3,1024,512],[3,3,512,1024]]
            for i in range(2):
                with tf.variable_scope('B5_'+str(i+2)):
                    model['B5_'+str(i+2)] = nn.ResUnit_full_dilate_2convs(self._data_format, model['B5_'+str(i+1)],
                                                      feed_dict, shape_dict['B5_2_3'],
                                                      var_dict=var_dict)

            # B6: [H/8, W/8, 1024] -> [H/8, W/8, 2048]
            shape_dict['B6'] = [[1,1,1024,512],[3,3,512,1024],[1,1,1024,2048]]
            with tf.variable_scope('B6'):
                model['B6'] = nn.ResUnit_hybrid_dilate_3conv(self._data_format, model['B5_3'],
                                                                 feed_dict,
                                                                 shape_dict['B6'],
                                                                 dropout=dropout,
                                                                 var_dict=var_dict)
            # B7: [H/8, W/8, 2048] -> [H/8, W/8, 4096]
            shape_dict['B7'] = [[1,1,2048,1024],[3,3,1024,2048],[1,1,2048,4096]]
            with tf.variable_scope('B7'):
                model['B7'] = nn.ResUnit_hybrid_dilate_3conv(self._data_format, model['B6'],
                                                                 feed_dict,
                                                                 shape_dict['B7'],
                                                                 dropout=dropout,
                                                                 var_dict=var_dict)

        # The wt unique part:
        # Gating operation, need semantic GT while training and inference
        model['gated_feat'] = self._gate(sem_gt, model['B7'])

        with tf.variable_scope("graddir"):
            # Further feature extractors
            shape_dict['grad_convs1'] = [[3,3,4096,512],[3,3,512,512]]
            with tf.variable_scope('convs1'):
                #model['wt_convs1'] = nn.ResUnit_tail(self._data_format, model['gated_feat'], feed_dict,
                #                                     shape_dict['wt_convs1'], var_dict)
                model['grad_convs1'] = nn.grad_convs(self._data_format, model['gated_feat'], feed_dict,
                                                     shape_dict['grad_convs1'], var_dict)
            # The normalize the output as the magnitued of GT
            shape_dict['grad_convs2'] = [[1,1,512,256],[1,1,256,256],[1,1,256,2]]
            with tf.variable_scope('convs2'):
                model['grad_convs2'] = nn.grad_norm(self._data_format, model['grad_convs1'], feed_dict,
                                                    shape_dict['grad_convs2'], var_dict)
            # Normalize the output to have unit vectors
            #model['grad_norm'] = nn.norm(self._data_format, model['grad_convs2'])

        with tf.variable_scope('wt'):
            # Further featrue extractors
            shape_dict['wt_convs1'] = [3,3,2,64]
            with tf.variable_scope('convs1'):
                model['wt_convs1'] = nn.WT_B0(self._data_format, model['grad_convs2'], feed_dict,
                                             shape_dict['wt_convs1'], var_dict)
            shape_dict['wt_convs2'] = [[3,3,64,128],[3,3,128,128],[1,1,64,128]]
            with tf.variable_scope('convs2'):
                model['wt_convs2'] = nn.WT_Block(self._data_format, model['wt_convs1'], feed_dict,
                                             shape_dict['wt_convs2'], var_dict)
            shape_dict['wt_convs3'] = [[3,3,128,256],[3,3,256,256],[1,1,128,256]]
            with tf.variable_scope('convs3'):
                model['wt_convs3'] = nn.WT_Block(self._data_format, model['wt_convs2'], feed_dict,
                                              shape_dict['wt_convs3'], var_dict)
            shape_dict['wt_convs4'] = [[3,3,256,512],[3,3,512,512],[1,1,256,512]]
            with tf.variable_scope('convs4'):
                model['wt_convs4'] = nn.WT_Block(self._data_format, model['wt_convs3'], feed_dict,
                                              shape_dict['wt_convs4'], var_dict)
            shape_dict['wt_tail'] = [[3,3,512,512],[3,3,512,16]]
            with tf.variable_scope('tail'):
                model['wt_tail'] = nn.WT_tail(self._data_format, model['wt_convs4'], feed_dict,
                                              shape_dict['wt_tail'], var_dict)

        return model

    def _gate(self, sem_input, feat_input):
        ''' This function takes inputs as semantic result and feature maps,
            returns gated feature maps, where non-relevant classes on the
            feature maps are set to zero
            Input: sem_input [batch_size, 1, H, W]
                   feat_input [batch_size, 4096, H, W]
            Output: gated feature maps [batch_size, H, W, 4096]
            NOTE: The default data format is "NCHW", the function also works for "NHWC" without any changes in code.
        '''

        # NOTE: Only gate class car for now
        sem_bool = tf.equal(sem_input, 13) #NOTE [batch_size, 1, H, W]
        sem_bin = tf.cast(sem_bool, tf.float32) #NOTE [batch_size, 1, H, W], zeros/ones

        ## Gate
        gated_feat = tf.multiply(feat_input, sem_bin) #NOTE [batch_size, 4096, H, W]

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

    def train_wt(self, image, sem_gt, wt_gt, params):
        '''This function only trains wt branch.
            Input: Image [batch_size, 512, 1024, 3]
                    sem_gt [batch_size, 64, 128, 1]
                    wt_gt [batch_size, 64, 128, 2], tf.float32
                        * 1st is the discretized watershed transforms
                        * 2nd is the weights for each discretization
                    params: decay_rate, lr, batch_size
        '''

        ## Get prediction prepared
        model = self._build_model(image, sem_gt, is_train=True)
        pred = model['wt_tail'] #NOTE pred  [batch_size, 16, 64,128] if "NCHW"

        ### TFBoard summary
        summ_pred = tf.argmax(tf.nn.softmax(tf.transpose(pred, [0, 2, 3, 1])), axis=3)
        summ_pred = tf.expand_dims(summ_pred, axis=-1)
        summ_pred = tf.cast(summ_pred, tf.float16)
        pred_sum_op = tf.summary.image('pred_out', summ_pred)
        ###

        pred = tf.reshape(pred, [params['batch_size'], 16, 64*128]) #NOTE: pred [batch_size, 16, 64*128] if "NCHW"
        if self._data_format == 'NCHW':
            pred = tf.transpose(pred, [0, 2, 1]) #NOTE, pred [batch_size, 64*128, 16]

        ## Get GTs prepared
        sem_gt = tf.reshape(sem_gt, [params['batch_size'],64*128]) #NOTE sem_gt [batch_size, 64*128]
        wt_label = tf.reshape(wt_gt[:,:,:,0:1], [params['batch_size'], 64*128]) #NOTE wt_label [batch_size, 64*128]
        wt_label = tf.cast(wt_label, tf.int32)
        wt_weight = tf.reshape(wt_gt[:,:,:,1:2], [params['batch_size'], 64*128]) #NOTE wt_weight [batch_size, 64*128]


        ## Get valid pixels to compute
        bool_mask = tf.equal(sem_gt,13) #NOTE bool_mask [batch_size, 64*128]
        valid_pred = tf.boolean_mask(pred, bool_mask)
        valid_wt_label = tf.boolean_mask(wt_label, bool_mask)
        valid_wt_weight = tf.boolean_mask(wt_weight, bool_mask)


        ## Compute weighted loss
        # if no label is 13, set loss to 0.0
        weighted_cross = tf.cond(tf.equal(tf.reduce_sum(tf.cast(bool_mask, tf.int32)), 0),
                                 lambda: 0.0,
                                 lambda: self._weight_cross_loss(valid_pred, valid_wt_label,valid_wt_weight))
        loss_total = weighted_cross + self._weight_decay(params['decay_rate'])

        ## Minimize
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(params['lr']).minimize(loss_total)

        return train_step, loss_total

    def _weight_cross_loss(self, valid_pred, valid_wt_label, valid_wt_weight):

        cross_out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_wt_label, logits=valid_pred) #NOTE same shape as valid_wt_label/valid_wt_weight
        # weighted_cross = tf.reduce_mean(tf.multiply(cross_out, valid_wt_weight))
        weighted_cross = tf.reduce_mean(cross_out)

        return weighted_cross

    def train_grad(self, image, sem_gt, grad_gt, params):
        ''' This function only trains graddir branch.
            Input: Image [batch_size, 512, 1024, 3]
                   sem_gt [batch_size, 64, 128, 1]
                   grad_gt [batch_size, 64, 128, 2]
                   params: decay_rate, lr
        '''

        model = self._build_model(image, sem_gt, is_train=True)
        pred = model['grad_norm'] #NOTE pred  [batch_size, 2, 64,128] if "NCHW"
        if self._data_format == 'NCHW':
            pred = tf.transpose(pred, [0, 2, 3, 1]) #NOTE, pred is [batch_size, 64, 128, 2]

        # The predicted graddir and GT are already normalized
        product = tf.reduce_sum(tf.multiply(pred,grad_gt), axis=3) #NOTE product [batch_size, 64,128]
        product = tf.maximum(product, -0.99)
        product = tf.minimum(product, 0.99)
        cos_out = tf.acos(product)
        sem_gt = tf.reshape(sem_gt, [params['batch_size'],64,128]) #NOTE sem_gt [batch_size, 64,128]
        bool_mask = tf.equal(sem_gt,13) #NOTE bool_mask [batch_size, 64,128]
        # if no label is 13, set loss to 0.0
        valid_cos_out = tf.cond(tf.equal(tf.reduce_sum(tf.cast(bool_mask, tf.int32)), 0), lambda: 0.0, lambda: tf.boolean_mask(cos_out, bool_mask))
        loss_grad = tf.reduce_mean(tf.square(valid_cos_out))
        loss_grad_valid = tf.cond(tf.is_nan(loss_grad), lambda: 0.0, lambda: loss_grad)
        loss_total = loss_grad_valid + self._weight_decay(params['decay_rate'])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(params['lr']).minimize(loss_total)

        return train_step, loss_total

    def inf(self, image, sem_gt):
        ''' Input: Image [batch_size, 1024, 2048, 3]
                   sem_gt [batch_size, 1024, 2048, 1]
            Output: upsampled wt result [batch_size, 1024, 2048, 16]
        '''
        small_size = [128, 256]
        sem_gt0 = tf.image.resize_images(sem_gt, small_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        model = self._build_model(image, sem_gt0, is_train=False)

        pred = model['wt_convs1'] #NOTE, pred is [batch_size, 16, 128, 256] if "NCHW"
        if self._data_format == 'NCHW':
            pred = tf.transpose(pred, [0, 2, 3, 1]) #NOTE, pred is [batch_size, 128, 256, 16]
        pred = self._upsample(pred, [1024,2048]) # [batch_size, 1024, 2048, 16]
        pred_label = tf.argmax(tf.nn.softmax(pred), axis=3)
        pred_label = tf.expand_dims(pred_label, axis=-1)
        pred_label = tf.cast(pred_label, tf.float32)

        # gate
        pred_label = self._gate(sem_gt, pred_label) # [batch_size, 1024, 2048, 1]
        pred_label = tf.cast(pred_label, tf.int32)

        return pred_label


