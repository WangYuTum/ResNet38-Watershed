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
        self._weight_dict = dt.load_weight(weight_path)
        self._var_dict = {}

        self._train_flag = {}
        self._train_flag['sem_train'] = False
        self._train_flag['grad_train'] = False

        self._num_classes = params.get('num_classes', 19)

    def _build_model(self, image, is_train=False, sem_train=False, grad_train=False):
        '''If is_train, save weight to self._var_dict,
           otherwise, don't save weights
           '''
        ## NOTE:
            # This unified network uses sharable ResNet36 conv layers which are freezed.
            # is_train indicates whether in train or inf mode
            # sem_train indicates training semantic branch
            # grad_train indicates training grad branch

        self._train_flag['sem_train'] = sem_train
        self._train_flag['grad_train'] = grad_train

        model = {}
        feed_dict = self._weight_dict
        if is_train:
            var_dict = self._var_dict
            if self._train_flag['sem_train'] != self._train_flag['grad_train']:
                continue
            else:
                sys.exit('Error in training mode: sem_train {0}, grad_train {1}'.format(self._train_flag['sem_train'], self._train_flag['grad_train']))
        else:
            var_dict = None
            self._train_flag['sem_train'] = False
            self._train_flag['grad_train'] = False

        if is_train:
            dropout = True
        else:
            dropout = False

        shape_dict = {}
        shape_dict['B0'] = [3,3,3,64]

        # The sharable conv features: ?? conv layers
        with tf.variable_scope('shared'):
            # B0: [H,W,3] -> [H,W,64]
            with tf.variable_scope('B0'):
                model['B0'] = nn.conv_layer(image, feed_dict, 1, 'SAME',
                                            shape_dict['B0'], var_dict, self._train_flag)

            # B2_1: [H,W,64] -> [H/2, W/2, 128]
            shape_dict['B2'] = {}
            shape_dict['B2']['side'] = [1,1,64,128]
            shape_dict['B2']['convs'] = [[3,3,64,128],[3,3,128,128]]
            with tf.variable_scope('B2_1'):
                model['B2_1'] = nn.ResUnit_downsample_2convs(model['B0'],
                                                             feed_dict,
                                                             shape_dict['B2'],
                                                             var_dict=var_dict, train_flag=self._train_flag)
            # B2_2, B2_3: [H/2, W/2, 128]
            for i in range(2):
                with tf.variable_scope('B2_'+str(i+2)):
                    model['B2_'+str(i+2)] = nn.ResUnit_2convs(model['B2_'+str(i+1)], feed_dict,
                                                              shape_dict['B2']['convs'][1],
                                                              var_dict=var_dict, train_flag=self._train_flag)

            # B3_1: [H/2, W/2, 128] -> [H/4, W/4, 256]
            shape_dict['B3'] = {}
            shape_dict['B3']['side'] = [1,1,128,256]
            shape_dict['B3']['convs'] = [[3,3,128,256],[3,3,256,256]]
            with tf.variable_scope('B3_1'):
                model['B3_1'] = nn.ResUnit_downsample_2convs(model['B2_3'],
                                                             feed_dict,
                                                             shape_dict['B3'],
                                                             var_dict=var_dict, train_flag=self._train_flag)
            # B3_2, B3_3: [H/4, W/4, 256]
            for i in range(2):
                with tf.variable_scope('B3_'+str(i+2)):
                    model['B3_'+str(i+2)] = nn.ResUnit_2convs(model['B3_'+str(i+1)], feed_dict,
                                                              shape_dict['B3']['convs'][1],
                                                              var_dict=var_dict, train_flag=self._train_flag)
            # B4_1: [H/4, W/4, 256] -> [H/8, W/8, 512]
            shape_dict['B4'] = {}
            shape_dict['B4']['side'] = [1,1,256,512]
            shape_dict['B4']['convs'] = [[3,3,256,512],[3,3,512,512]]
            with tf.variable_scope('B4_1'):
                model['B4_1'] = nn.ResUnit_downsample_2convs(model['B3_3'],
                                                                 feed_dict,
                                                                 shape_dict['B4'],
                                                                 var_dict=var_dict, train_flag=self._train_flag)
            # B4_2 ~ B4_6: [H/8, W/8, 512]
            for i in range(5):
                with tf.variable_scope('B4_'+str(i+2)):
                    model['B4_'+str(i+2)] = nn.ResUnit_2convs(model['B4_'+str(i+1)],
                                                                   feed_dict,
                                                                   shape_dict['B4']['convs'][1],
                                                                   var_dict=var_dict, train_flag=self._train_flag)
            # B5_1: [H/8, W/8, 512] -> [H/8, W/8, 1024]
            shape_dict['B5_1'] = {}
            shape_dict['B5_1']['side'] = [1,1,512,1024]
            shape_dict['B5_1']['convs'] = [[3,3,512,512],[3,3,512,1024]]
            with tf.variable_scope('B5_1'):
                model['B5_1'] = nn.ResUnit_hybrid_dilate_2conv(model['B4_6'],
                                                                   feed_dict,
                                                                   shape_dict['B5_1'],
                                                                   var_dict=var_dict, train_flag=self._train_flag)
            # B5_2, B5_3: [H/8, W/8, 1024]
            # Shape for B5_2, B5_3
            shape_dict['B5_2_3'] = [[3,3,1024,512],[3,3,512,1024]]
            for i in range(2):
                with tf.variable_scope('B5_'+str(i+2)):
                    model['B5_'+str(i+2)] = nn.ResUnit_full_dilate_2convs(model['B5_'+str(i+1)],
                                                      feed_dict, shape_dict['B5_2_3'],
                                                      var_dict=var_dict, train_flag=self._train_flag)

            # B6: [H/8, W/8, 1024] -> [H/8, W/8, 2048]
            shape_dict['B6'] = [[1,1,1024,512],[3,3,512,1024],[1,1,1024,2048]]
            with tf.variable_scope('B6'):
                model['B6'] = nn.ResUnit_hybrid_dilate_3conv(model['B5_3'],
                                                                 feed_dict,
                                                                 shape_dict['B6'],
                                                                 dropout=dropout,
                                                                 var_dict=var_dict, train_flag=self._train_flag)
            # B7: [H/8, W/8, 2048] -> [H/8, W/8, 4096]
            shape_dict['B7'] = [[1,1,2048,1024],[3,3,1024,2048],[1,1,2048,4096]]
            with tf.variable_scope('B7'):
                model['B7'] = nn.ResUnit_hybrid_dilate_3conv(model['B6'],
                                                                 feed_dict,
                                                                 shape_dict['B7'],
                                                                 dropout=dropout,
                                                                 var_dict=var_dict, train_flag=self._train_flag)

        # The semantic unique part: conv1(+bias1) + conv2(+bias2). Output feature stride=8
        with tf.variable_scope('semantic'):
            shape_dict['semantic'] = [[3,3,4096,512],[3,3,512,self._num_classes]]
            model['semantic'] = nn.ResUnit_tail(model['B7'], feed_dict,
                                            shape_dict['semantic'], var_dict, train_flag=self._train_flag)

        # The graddir unique part: conv1 + conv2 + 3*conv3(kernel: [1x1])
        # Gating operation, need result from semantic branch
        gated_feat = self._gate(model['semantic'], model['B7'])

        with tf.variable_scope("graddir"):
            # The normal feature layers
            shape_dict['grad_convs1'] = [[3,3,4096,512],[3,3,512,512]]
            with tf.variable_scope('convs'):
                model['grad_convs1'] = nn.grad_convs(gated_feat, feed_dict,
                                                   shape_dict['grad_convs1'], var_dict, train_flag=self._train_flag)
            # The norm layers to normalize the output to have same magnitude as grad GT
            shape_dict['grad_convs2'] = [[1,1,512,256],[1,1,256,128],[1,1,128,2]]
            with tf.variable_scope('norm'):
                model['grad_convs2'] = nn.grad_norm(model['grad_convs1'], feed_fict,
                                                 shape_dict['grad_convs2'], var_dicti, train_flag=self._train_flag)
            # Normalize the output to have unit vectors
            model['grad_norm'] = nn.norm(model['grad_convs2'])

        return model

    def _gate(self, sem_input, feat_input):
        ''' This function takes inputs as semantic result and feature maps,
            returns gated feature maps, where non-relevant classes on the
            feature maps are set to zero
            Input: sem_input [1, H, W, 19]
                   feat_input [1, H, W, 4096]
            Output: gated feature maps [1, H, W, 4096]'''

        sem_out = tf.argmax(tf.nn.softmax(sem_input), axis=3)
        # TODO: Only gate class car, car label
        sem_bool = tf.equal(sem_out, 13)
        sem_bin = tf.cast(sem_bool, tf.float32)

        gated_feat = tf.multiply(feat_input, sem_bin)

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

    ## TODO
    def train(self, image, label, params):
        '''Input: Image [batch_size, H, W, C]
                  Label [batch_size]
                  params: 'num_class', 'batch_size', 'decay_rate' '''

        # Here padding to 36x36 then randomly crop per image.
        padded_img = tf.image.resize_image_with_crop_or_pad(image, 36, 36)
        cropped_img = tf.random_crop(padded_img, [params['batch_size'], 32, 32,3])
        # padding and cropping end

        # Here randomly flip each image
        flipped_img = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped_img)
        # randomly flipping end

        model = self._build_model(flipped_img, is_train=True)
        prediction = model['fc_out']
        label = tf.reshape(label, [params['batch_size']])

        # compute train accuracy
        pred_label = tf.argmax(tf.nn.softmax(prediction), axis=1)
        correct_pred_bools = tf.equal(pred_label, label)
        correct_preds = tf.reduce_sum(tf.cast(correct_pred_bools, tf.float32))
        train_acc = correct_preds/params['batch_size']
        entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,
                                                                      logits=prediction))
        total_loss = entropy_loss + self._weight_decay(params['decay_rate'])
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # default learning rate for Adam: 0.001
            train_op = tf.train.AdamOptimizer().minimize(total_loss)

        return train_op, total_loss, train_acc, correct_preds

    def train_sem(self, image, label, params):
        ''' This function only trains semantic branch.
            Input: Image [1, H, W, 3]
                   Label [1, H*W]
                   params: decay_rate, lr
        '''
        # NOTE: train on downsampled results 

        model = self._build_model(image, is_train=True, sem_train=True, grad_train=False)
        pred = model['semantic']

        old_shape = tf.shape(pred)
        new_shape = [old_shape[0], old_shape[1]*old_shape[2], self._num_classes]
        pred = tf.reshape(pred, new_shape)

        # TODO: check void label number
        void_bool = tf.equal(label, 19)
        valid_bool = tf.logical_not(void_bool)
        valid_label = tf.boolean_mask(label, valid_bool)
        valid_pred = tf.boolean_mask(pred, valid_bool)

        loss_sem = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=valid_label, logits=valid_pred))
        loss_total = loss_sem + self._weight_decay(params['decay_rate'])
        # Don't update BN parameters, use normal optimizer
        train_step = tf.train.AdamOptimizer(params['lr']).minimize(loss_total)

        return train_step, loss_total

    def train_grad(self, image, label, params):
        '''This function only trains graddir branch.
            Input: Image [1, H, W, 3]
                   Label [1, H, W, 2]
                   params: decay_rate, lr
        '''
        # NOTE: train on downsampled results

        model = self._build_model(image, is_train=True, sem_train=False, grad_train=True)
        pred = model['grad_norm']

        # remove the first dim
        pred = tf.squeeze(pred)
        label = tf.squeeze(label)

        # The predicted graddir and GT are already normalized
        product = tf.reduce_sum(tf.multiply(pred,label), axis=2)
        cos_out = tf.acos(product)
        loss_grad = tf.square(tf.norm(cos_out))
        loss_total = loss_grad + self._weight_decay(params['decay_rate'])

        # Don't update BN parameters, use normal optimizer
        train_step = tf.train.AdamOptimizer(params['lr']).minimize(loss_total)

        return train_step, loss_total


    def inf_grad(self, image):
        ''' Input: Image [1, H, W, C]
            Output: upsampled semantic result [1, 1024, 2048]
                    upsampled graddir result [1, 1024, 2048, 2]'''

        model = self._build_model(image, is_train=False)
        pred_sem = model['semantic']
        pred_grad = model['grad_norm']
        upsampled_sem = self._upsample(pred_sem, [1024,2048])
        upsampled_grad = self._upsample(pred_grad, [1024,2048])

        ## Predict class label: [batch, H, W, C]
        pred_sem_label = tf.argmax(tf.nn.softmax(upsampled_sem), axis=3)

        return pred_sem_label, upsampled_grad

    def inf_semantic(self, image):
        ''' Input: Image [1, H, W, C]
            Output: upsampled pred [1, 1024, 2048, 1]
            Note that it only produce semantic output.'''

        model = self._build_model(image, is_train=False)
        pred = model['semantic']
        upsampled_pred = self._upsample(pred, [1024,2048])

        ## Predict class label: [batch, H, W, C]
        pred_label = tf.argmax(tf.nn.softmax(upsampled_pred), axis=3)

        return pred_label



