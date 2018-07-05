from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
sys.path.append("..")
import numpy as np
from scipy.misc import imsave
from PIL import Image
import tensorflow as tf
import data_utils as dt
from core import resnet38
from eval import evalPixelSemantic

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#config_gpu = tf.ConfigProto()
#config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.9
test_data_params = {'mode': 'test_final',
                     'batch_size': 1}

# The data pipeline should be on CPU
with tf.device('/cpu:0'):
    CityData = dt.CityDataSet(test_data_params)
    next_batch = CityData.next_batch()

# Hparameter
model_params = {'num_classes': 19,
                'feed_weight': '../data/saved_weights/final_adam_batch2/watershed_final8s_ep18.npy',
                'batch_size': 1,
                'data_format': "NCHW", # optimal for cudnn
                }

num_val = 500
num_test = 1525
num_train = 2975
iterations = 1
#iterations = int(num_train / model_params['batch_size'])
batch = model_params['batch_size']

res38 = resnet38.ResNet38(model_params)
# Test on single image
test_img = Image.open('../data/CityDatabase/leftImg8bit/test/berlin/berlin_000004_000019_leftImg8bit.png')
test_img = np.array(test_img, dtype=np.float32)
test_img *= (1.0/255)
mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
test_img -= mean
std = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
test_img /= std
test_img = test_img[np.newaxis,:]
test_tensor = tf.convert_to_tensor(test_img, dtype=tf.float32)
[sem_label, wt_label, sem_prob] = res38.inf(image=test_tensor)
# Test on whole test_set
# [sem_label, wt_label, sem_prob] = res38.inf(image=next_batch['img'])
init = tf.global_variables_initializer()

with tf.Session() as sess:
#with tf.Session(config=config_gpu) as sess:

    sess.run(init)
    print('Finished building inference network ResNet38-8s')

    print('Start inference...')
    for i in range(iterations):
        print('iter {0}:'.format(i))
        [sem_label_, wt_label_, sem_prob_] = sess.run([sem_label, wt_label, sem_prob])
        pred_sem_label = np.squeeze(sem_label_) # [batch_size, 1024, 2048]
        pred_wt_label = np.squeeze(wt_label_) # [batch_size, 1024, 2048]
        CityData.save_trainID_img(pred_sem_label, pred_wt_label, True, True)
        #TODO: save sem_prob_

    print("Inference done! Start transforming to colored ...")
    CityData.pred_to_color()
    #print("Start transforming to labelIDs ...")
    #CityData.pred_to_labelID()
    #print("Transform done! Ready to generate evaluation files.")
    #print("Start evaluating semantic accuracy ...")
    #accuracy = evalPixelSemantic.run_eval('../data/sempred')
