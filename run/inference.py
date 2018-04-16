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
from eval import evalPixelSemantic

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.9
test_data_params = {'mode': 'test_final', #NOTE: later change to val_semgrad
                     'batch_size': 1}

# The data pipeline should be on CPU
with tf.device('/cpu:0'):
    CityData = dt.CityDataSet(test_data_params)
    next_batch = CityData.next_batch()

# Hparameter
model_params = {'num_classes': 19,
                'feed_weight': '../data/saved_weights/WTN_adam_batch3/watershed_pre-semgrad-wt_wtn_ep9.npy',
                'batch_size': 1,
                'data_format': "NCHW", # optimal for cudnn
                }

num_val = 500
num_test = 1525
iterations = num_test
batch = model_params['batch_size']

res38 = resnet38.ResNet38(model_params)
predict = res38.inf(image=next_batch['img'], sem_gt=next_batch['sem_gt'])
init = tf.global_variables_initializer()

# with tf.Session() as sess:
with tf.Session(config=config_gpu) as sess:

    sess.run(init)
    print('Finished building inference network ResNet38-8s')

    print('Start inference...')
    for i in range(iterations):
        print('iter {0}:'.format(i))
        pred_out = sess.run(predict) #NOTE: [batch_size, 1024, 2048, 1], discretized output

        for j in range(batch):
            pred_img = np.squeeze(pred_out) # [1, 1024, 2048]
            # print('Save pred to {0}'.format("pred_wt"+str(i*batch+j)+".npy"))
            # imsave("pred_grad%d.png"%(i*batch+j), pred_img) # Rescales image to 0-255
            # save to numpy array, to avoid rescale
            np.save("pred_grad%d"%(i*batch+j), pred_img)

