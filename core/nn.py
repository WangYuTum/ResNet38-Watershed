from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def ResUnit_downsample_2convs(data_format, input_tensor, feed_dict, shape_dict,
                              var_dict=None):
    '''One of the Residual Block which performs downsampling.'''

    if var_dict is not None:
        is_train = True
    else:
        is_train = False

    # The first bath norm layer
    BN_out1 = BN(data_format, input_tensor=input_tensor, feed_dict=feed_dict,
                 bn_scope='bn1', is_train=is_train, shape=shape_dict['convs'][0][2],
                 var_dict=var_dict)
    RELU_out1 = ReLu_layer(BN_out1)

    # The shortcut convolution layer, using stride=2 to perform downsample
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, RELU_out1, feed_dict, 2, 'SAME',
                              shape_dict['side'], var_dict)
    # Downsampled convolution layer, using stride=2 to perform downsample
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_layer(data_format, RELU_out1, feed_dict, 2, 'SAME',
                               shape_dict['convs'][0], var_dict)
    # The second batch norm layer
    BN_out2 = BN(data_format, CONV_out1, feed_dict, 'bn2',is_train, shape_dict['convs'][1][2], var_dict)
    RELU_out2 = ReLu_layer(BN_out2)
    # The second convolution layer
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_layer(data_format, RELU_out2, feed_dict, 1, 'SAME',
                               shape_dict['convs'][1], var_dict)
    # Fuse
    ResUnit_out = tf.add(side_out, CONV_out2)

    return ResUnit_out

def ResUnit_hybrid_dilate_2conv(data_format, input_tensor, feed_dict, shape_dict, var_dict):
    '''The layer for B5_1'''

    if var_dict is not None:
        is_train = True
    else:
        is_train = False

    # The first bath norm layer
    BN_out1 = BN(data_format, input_tensor=input_tensor, feed_dict=feed_dict,
                 bn_scope='bn1', is_train=is_train,
                 shape=shape_dict['convs'][0][2], var_dict=var_dict)
    RELU_out1 = ReLu_layer(BN_out1)

    # The side convolution layer, no downsampling
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, RELU_out1, feed_dict, 1, 'SAME',
                              shape_dict['side'], var_dict)
    # This first convolution layer
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_layer(data_format, RELU_out1, feed_dict, 1, 'SAME',
                               shape_dict['convs'][0], var_dict)
    # The second batch norm layer
    BN_out2 = BN(data_format, CONV_out1, feed_dict, 'bn2', is_train,
                 shape_dict['convs'][1][2], var_dict)
    RELU_out2 = ReLu_layer(BN_out2)
    # dilated convolution layer
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_dilate_layer(data_format, RELU_out2, feed_dict, 2, 'SAME',
                                  shape_dict['convs'][1], var_dict)

    # Fuse
    ResUnit_out = tf.add(side_out, CONV_out2)

    return ResUnit_out

def ResUnit_full_dilate_2convs(data_format, input_tensor, feed_dict, shape_dict,
                               var_dict=None):
    '''Residul Unit: all convolution layers are dilated. For B5_2, B5_3'''

    if var_dict is not None:
        is_train = True
    else:
        is_train = False

    # The first batch norm layer
    BN_out1 = BN(data_format, input_tensor=input_tensor, feed_dict=feed_dict,
                 bn_scope='bn1', is_train=is_train,
                 shape=shape_dict[0][2], var_dict=var_dict)
    RELU_out1 = ReLu_layer(BN_out1)
    # The first dilated convolution layer
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_dilate_layer(data_format, RELU_out1, feed_dict, 2, 'SAME', shape_dict[0], var_dict)
    # The second batch norm layer
    BN_out2 = BN(data_format, CONV_out1, feed_dict, 'bn2', is_train,
                             shape_dict[1][2], var_dict)
    RELU_out2 = ReLu_layer(BN_out2)
    # The second dilated convolution layer
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_dilate_layer(data_format,RELU_out2, feed_dict, 2, 'SAME', shape_dict[1], var_dict)

    # Fuse
    ResUnit_out = tf.add(input_tensor, CONV_out2)

    return ResUnit_out

def ResUnit_hybrid_dilate_3conv(data_format, input_tensor, feed_dict, shape_dict, dropout,
                                var_dict):
    '''Residual Unit with 3 convolution layers(including 1 dilated conv). For
    B6, B7'''

    if var_dict is not None:
        is_train = True
    else:
        is_train = False

    # The first batch norm layer
    BN_out1 = BN(data_format, input_tensor=input_tensor, feed_dict=feed_dict,
                 bn_scope='bn1', is_train=is_train,
                 shape=shape_dict[0][2], var_dict=var_dict)
    RELU_out1 = ReLu_layer(BN_out1)
    # side conv
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, RELU_out1, feed_dict, 1,
                              'SAME', shape_dict[2], var_dict)
    # The first conv layer
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_layer(data_format, RELU_out1, feed_dict, 1, 'SAME', shape_dict[0], var_dict)
    # The second batch norm layer
    BN_out2 = BN(data_format, CONV_out1, feed_dict, 'bn2', is_train, shape_dict[1][2],
                 var_dict)
    RELU_out2 = ReLu_layer(BN_out2)
    # The second conv layer - dilated
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_dilate_layer(data_format, RELU_out2, feed_dict,  4,
                                      'SAME', shape_dict[1], var_dict)

    scope_name = tf.get_variable_scope().name
    if dropout:
        if scope_name == 'B7':
            CONV_out2 = tf.nn.dropout(CONV_out2, 0.7)
            print('Dropout on {0} with rate {1}'.format(scope_name+'/conv2', 0.3))
    # The third batch norm layer
    BN_out3 = BN(data_format, CONV_out2, feed_dict, 'bn3', is_train, shape_dict[2][2],
                 var_dict)
    RELU_out3 = ReLu_layer(BN_out3)
    # The third conv layer
    with tf.variable_scope('conv3'):
        CONV_out3 = conv_layer(data_format, RELU_out3, feed_dict,  1, 'SAME',
                               shape_dict[2], var_dict)
    if dropout:
        if scope_name == 'B6':
            CONV_out3 = tf.nn.dropout(CONV_out3, 0.7)
            print('Dropout on {0} with rate {1}'.format(scope_name+'/conv3', 0.3))
        if scope_name == 'B7':
            CONV_out3 = tf.nn.dropout(CONV_out3, 0.5)
            print('Dropout on {0} with rate {1}'.format(scope_name+'/conv3', 0.5))
    # Fuse
    ResUnit_out = tf.add(side_out, CONV_out3)

    return ResUnit_out

def ResUnit_tail(data_format, input_tensor, feed_dict, shape_dict, var_dict=None):
    '''The ResNet38 Tail for semantic segmantation'''

    if var_dict is not None:
        is_train = True
    else:
        is_train = False

    # The batch norm layer
    BN_out1 = BN(data_format, input_tensor=input_tensor, feed_dict=feed_dict,
                 bn_scope='bn1', is_train=is_train,
                 shape=shape_dict[0][2], var_dict=var_dict)
    RELU_out1 = ReLu_layer(BN_out1)

    # The first dilated conv layer
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_dilate_layer(data_format, RELU_out1, feed_dict, 12,
                                      'SAME', shape_dict[0], var_dict)
    with tf.variable_scope('bias1'):
        BIAS_out1 = bias_layer(data_format, CONV_out1, feed_dict, shape_dict[0][3], var_dict)
    RELU_out2 = ReLu_layer(BIAS_out1)
    # The second dilated conv layer
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_dilate_layer(data_format, RELU_out2, feed_dict, 12,
                                      'SAME', shape_dict[1], var_dict)
    with tf.variable_scope('bias2'):
        BIAS_out2 = bias_layer(data_format, CONV_out2, feed_dict, shape_dict[1][3], var_dict)
    return BIAS_out2

def ResUnit_2convs(data_format, input_tensor, feed_dict, shape, var_dict=None):
    '''Standard Residual Unit without downsampling'''

    if var_dict is not None:
        is_train = True
    else:
        is_train = False

    # The first batch norm layer
    BN_out1 = BN(data_format, input_tensor=input_tensor, feed_dict=feed_dict,
                 bn_scope='bn1', is_train=is_train, shape=shape[2],
                 var_dict=var_dict)
    RELU_out1 = ReLu_layer(BN_out1)
    # The first conv layer
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_layer(data_format, RELU_out1, feed_dict, 1, 'SAME', shape, var_dict)
    # The second batch norm layer
    BN_out2 = BN(data_format, CONV_out1, feed_dict, 'bn2', is_train, shape[2], var_dict)
    RELU_out2 = ReLu_layer(BN_out2)
    # The second conv layer
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_layer(data_format, RELU_out2, feed_dict, 1, 'SAME', shape, var_dict)

    # Fuse
    ResUnit_out = tf.add(input_tensor, CONV_out2)

    return ResUnit_out

def grad_convs(data_format, input_tensor, feed_dict, shape, var_dict=None):
    '''the conv stage for graddir branch'''

    if var_dict is not None:
        is_train = True
    else:
        is_train = False

    # the first batch norm layer
    BN_out1 = BN(data_format, input_tensor=input_tensor, feed_dict=feed_dict,
                 bn_scope='bn1', is_train=is_train, shape=shape[0][2], var_dict=var_dict)
    RELU_out1 = ReLu_layer(BN_out1)
    # The first conv layer
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_layer(data_format, RELU_out1, feed_dict, 1, 'SAME', shape[0], var_dict)
    # The second batch norm layer
    BN_out2 = BN(data_format, CONV_out1, feed_dict, 'bn2', is_train, shape[1][2], var_dict)
    RELU_out2 = ReLu_layer(BN_out2)
    # The second conv layer
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_layer(data_format, RELU_out2, feed_dict, 1, 'SAME', shape[1], var_dict)

    return CONV_out2

def grad_norm(data_format, input_tensor, feed_dict, shape, var_dict=None):
    '''the grad normalization stage'''

    if var_dict is not None:
        is_train = True
    else:
        is_train = False

    # The only relu layer
    RELU_out1 = ReLu_layer(input_tensor)
    # The first conv layer
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_layer(data_format, RELU_out1, feed_dict, 1, 'SAME', shape[0], var_dict)
    RELU_out2 = ReLu_layer(CONV_out1)
    # The second conv layer
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_layer(data_format, RELU_out2, feed_dict, 1, 'SAME', shape[1], var_dict)
    RELU_out3 = ReLu_layer(CONV_out2)
    # The third conv layer
    with tf.variable_scope('conv3'):
        CONV_out3 = conv_layer(data_format, RELU_out3, feed_dict, 1, 'SAME', shape[2], var_dict)

    return CONV_out3

def norm(data_format, input_tensor):
    ''' Normalize the gradddir
        Input: [batch_size, H, W, 2]
    '''

    if data_format == 'NCHW':
        norm_dim = 1
    else:
        norm_dim = 3

    ## use tensorflow implementation
    normed_tensor = tf.nn.l2_normalize(input_tensor, dim=norm_dim)

    return normed_tensor

def conv_dilate_layer(data_format, input_tensor, feed_dict, rate, padding='SAME',
                      shape=None, var_dict=None):
    '''dilated convolution layer using tensorflow atrous_conv2d'''

    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)
    kernel = get_conv_kernel(feed_dict, shape)

    # The TF atrous needs data format to be 'NHWC'
    if data_format == 'NCHW':
        # switch back to 'NHWC'
        input_tensor = tf.transpose(input_tensor, [0,2,3,1])
        conv_out = tf.nn.atrous_conv2d(input_tensor, kernel, rate, padding)
        conv_out = tf.transpose(conv_out, [0,3,1,2])
    else:
        conv_out = tf.nn.atrous_conv2d(input_tensor, kernel, rate, padding)

    if var_dict is not None:
        if not var_dict.has_key(scope_name):
            var_dict[scope_name] = {}
        var_dict[scope_name]['kernel'] = kernel

    return conv_out

def conv_layer(data_format, input_tensor, feed_dict, stride, padding='SAME', shape=None,
               var_dict=None):
    '''The standard convolution layer'''

    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)
    kernel = get_conv_kernel(feed_dict, shape)

    if data_format == "NCHW":
        feed_strides = [1,1,stride, stride]
    else:
        feed_strides = [1,stride, stride,1]
    conv_out = tf.nn.conv2d(input_tensor, kernel, strides=feed_strides, padding=padding, data_format=data_format)

    if var_dict is not None:
        if not var_dict.has_key(scope_name):
            var_dict[scope_name] = {}
        var_dict[scope_name]['kernel'] = kernel

    return conv_out

def BN(data_format, input_tensor, feed_dict=None, bn_scope=None,is_train=False, shape=None,
       var_dict=None):
    '''Using tensorflow implemented batch norm layer'''

    scope_name = tf.get_variable_scope().name
    with tf.variable_scope(bn_scope):
        [beta_init, gamma_init, moving_mean_init, moving_var_init] = get_bn_params(feed_dict=feed_dict, shape=shape)

    bn_training = True
    bn_trainable = True
    # if not training, fix all parameters
    if is_train is False:
        bn_training = False
        bn_trainable = False
        bn_fused=True
    # if training
    else:
        bn_fused=False
        # fix moving statistics and trainable variables in shared layers
        if scope_name.find('shared') != -1:
            bn_training = True
            bn_trainable = True
        else:
            bn_training = True
            bn_trainable = True
    if data_format == 'NCHW':
        norm_axis=1
    else:
        norm_axis=-1

    BN_out = tf.layers.batch_normalization(inputs=input_tensor,
                                          axis=norm_axis, # the normed axis, determined by data format
                                          momentum=0.99, # default
                                          epsilon=0.001, # default
                                          center=True, # use beta
                                          scale=True, # use gamma
                                          beta_initializer=beta_init,
                                          gamma_initializer=gamma_init,
                                          moving_mean_initializer=moving_mean_init,
                                          moving_variance_initializer=moving_var_init,
                                          beta_regularizer=None,
                                          gamma_regularizer=None,
                                          training=bn_training, # using current batch statistics(T) or moving statistics(F)
                                          trainable=bn_trainable, # Depends on whether train(T) or not(F)
                                          name=bn_scope, # current scope
                                          reuse=None,
                                          fused=bn_fused)
    if var_dict is not None:
        with tf.variable_scope(bn_scope, reuse=True):
            nested_scope = tf.get_variable_scope().name
            if not var_dict.has_key(nested_scope):
                var_dict[nested_scope] = {}
            var_dict[nested_scope]['beta'] = tf.get_variable('beta')
            var_dict[nested_scope]['gamma'] = tf.get_variable('gamma')
            var_dict[nested_scope]['moving_mean'] = tf.get_variable('moving_mean')
            var_dict[nested_scope]['moving_var'] = tf.get_variable('moving_variance')

    return BN_out

def bilinear_upscore_layer(input_tensor, new_size):
    '''
    Given input_tensor: [batch, H, W, C] and new_size: [new_H, new_W],
    Return: [batch, new_H, new_W, C]
    '''
    print('Layer name: Bilinear interpolation.')
    out = tf.image.resize_bilinear(input_tensor, new_size)

    return out

def get_conv_kernel(feed_dict, shape):
    '''Retrieve or create kernel weights for the current variable scope.'''

    scope_name = tf.get_variable_scope().name
    if not feed_dict.has_key(scope_name):
        print('No matched kernel for %s, randomly initialize with shape %s'%(scope_name, str(shape)))
        init = tf.truncated_normal_initializer(stddev=0.001)
    else:
        init_val = feed_dict[scope_name]['kernel']
        shape = init_val.shape
        print('Load kernel with shape %s'%str(shape))
        init = tf.constant_initializer(value=init_val)

    # during alternate training, fix variables in shared layers
    if scope_name.find('shared') != -1:
        var = tf.get_variable(name='kernel', initializer=init, shape=shape, trainable=True)
    else:
        var = tf.get_variable(name='kernel', initializer=init, shape=shape, trainable=True)

    return var

def ReLu_layer(relu_in):
    '''Standard relu operation'''

    relu_out = tf.nn.relu(relu_in)

    return relu_out

def get_bn_params(feed_dict, shape):
    '''Retrive batch norm params'''

    scope_name = tf.get_variable_scope().name
    if not feed_dict.has_key(scope_name):
        print('No matched BN params %s, randomly initialize BN params with shape %s'%(scope_name, str(shape)))
        init_beta = tf.zeros_initializer()
        init_mean = tf.zeros_initializer()
        init_gamma = tf.ones_initializer()
        init_var = tf.ones_initializer()
    else:
        beta = feed_dict[scope_name]['beta']
        gamma = feed_dict[scope_name]['gamma']
        moving_mean = feed_dict[scope_name]['moving_mean']
        moving_var = feed_dict[scope_name]['moving_var']
        print('Load BN params %s with shape %s'%(scope_name, str(shape)))
        init_beta = tf.constant_initializer(value=beta)
        init_gamma = tf.constant_initializer(value=gamma)
        init_mean = tf.constant_initializer(value=moving_mean)
        init_var = tf.constant_initializer(value=moving_var)

    return init_beta, init_gamma, init_mean, init_var

def bias_layer(data_format, input_tensor, feed_dict, shape, var_dict):

    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)
    bias = get_bias(feed_dict, shape)

    bias_out = tf.nn.bias_add(input_tensor, bias, data_format=data_format)
    if var_dict is not None:
        if not var_dict.has_key(scope_name):
            var_dict[scope_name] = {}
        var_dict[scope_name]['bias'] = bias

    return bias_out

def get_bias(feed_dict, shape):

    scope_name = tf.get_variable_scope().name
    if not feed_dict.has_key(scope_name):
        print('No matched bias for %s, randomly initialize with shape %s'%(scope_name, str(shape)))
        init = tf.truncated_normal_initializer(stddev=0.001)
    else:
        init_val = feed_dict[scope_name]['bias']
        shape = init_val.shape
        print('Load bias with shape: %s' % str(shape))
        init = tf.constant_initializer(value=init_val)

    # during alternate training, fix variables in shared layers
    if scope_name.find('shared') != -1:
        var = tf.get_variable(name="bias", initializer=init, shape=shape, trainable=True)
    else:
        var = tf.get_variable(name='kernel', initializer=init, shape=shape, trainable=True)

    return var
