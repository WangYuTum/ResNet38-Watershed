from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
import data_utils as dt
from core import resnet38

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.8

train_data_params = {'data_path': 'data/cifar-10-batches-py/',
                     'batch_size': 125,
                     'mode': 'Test'}
dataset = dt.CIFAR10(train_data_params)

params = {'batch_size': 125,
          'feed_path': 'data/saved_weights/modelA_40e3_130.npy'}

# with tf.Session() as sess:
with tf.Session(config=config_gpu) as sess:
    res38 = resnet38.ResNet38(params['feed_path'])
    batch_size = params['batch_size']

    # From here
    test_img = tf.placeholder(tf.float32, shape=[batch_size, 32, 32,
                                                  3])
    test_label = tf.placeholder(tf.int64, shape=[batch_size])

    # 10000 / 125 = 80 iters
    correct_preds = res38.inf(image=test_img, label=test_label, params=params)

    init = tf.global_variables_initializer()
    sess.run(init)

    num_iters = np.int32(10000 / batch_size)
    num_correct = 0.0
    print('Start inference...')
    for iters in range(num_iters):
        print('iter %d'%iters)
        next_images, next_labels = dataset.next_batch()
        test_feed_dict = {test_img: next_images, test_label: next_labels}
        correct_preds_ = sess.run(correct_preds, test_feed_dict)
        num_correct += correct_preds_
    acc = num_correct / 10000.0
    print('Test acc is %f'%acc)


