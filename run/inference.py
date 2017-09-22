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
                'feed_weight': '../data/saved_weights/watershed_preimgneta1_grad8s_up_ep3.npy'}
num_val = 500
num_test = 1525
iterations = 4

# with tf.Session() as sess:
with tf.Session(config=config_gpu) as sess:
    res38 = resnet38.ResNet38(model_params)

    # Feed test/val image, batch=1
    img = tf.placeholder(tf.float32, shape=[1, 1024, 2048, 3])
    # Get inference result
    predict = res38.inf(img)

    print('Finished building inference network ResNet38-8s-grad')
    init = tf.global_variables_initializer()
    sess.run(init)

    print('Start inference...')
    for i in range(iterations):
        print('iter {0}:'.format(i))
        next_pair = dataset.next_batch()
        next_pair_image = next_pair[0]

        print("shape of img: {0}".format(np.shape(next_pair_image)))
        feed_dict_ = {img: next_pair_image}

        pred_out = sess.run(predict, feed_dict=feed_dict_)
        pred_img = = np.concatenate((pred_out, np.zeros((1024,2048,2),dtype=np.float32)), axis=-1)
        print("Save pred to {0}".format("pred_grad"+str(i)+".png"))
        imsave("pred_grad%d.png"%i,pred_img[:,:,0:3])
