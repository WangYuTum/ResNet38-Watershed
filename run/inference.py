from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
sys.path.append("..")
import numpy as np
from scipy.misc import imsave
import tensorflow as tf
import data_utils as dt
from core import resnet38

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
test_data_params = {'data_dir': '../data/CityDatabase',
                     'dataset': 'val',
                     'batch_size': 1}

dataset = dt.CityDataSet(test_data_params)

model_params = {'num_classes': 19,
                'feed_weight': '../data/saved_weights/watershed_preimgneta1_grad8s2_up_ep9.npy'}
num_val = 500
num_test = 1525
iterations = 4

with tf.Session() as sess:
    res38 = resnet38.ResNet38(model_params)

    # Feed test/val image, batch=1
    test_img = tf.placeholder(tf.float32, shape=[1, 1024, 2048, 3])
    test_sem_gt = tf.placeholder(tf.int32, shape=[1, 1024, 2048])

    # Get inference result
    predict = res38.inf(image=test_img, sem_gt=test_sem_gt)

    print('Finished building inference network ResNet38-8s-grad')
    init = tf.global_variables_initializer()
    sess.run(init)

    print('Start inference...')
    for i in range(iterations):
        print('iter {0}:'.format(i))
        next_images, next_sem_gt, next_labels = dataset.next_batch() # images [batch_size,H,W,3], sem_gt [batch_size,H,W], labels [batch_size,H,W,2]
        feed_dict_ = {test_img: next_images, test_sem_gt: next_sem_gt}

        pred_out = sess.run(predict, feed_dict=feed_dict_)
        pred_img = np.concatenate((pred_out, np.zeros((1024,2048,2),dtype=np.float32)), axis=-1)
        print("Save pred to {0}".format("pred_grad"+str(i)+".png"))
        imsave("pred_grad%d.png"%i,pred_img[:,:,0:3])
