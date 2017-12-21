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
#config_gpu = tf.ConfigProto()
#config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.9
test_data_params = {'mode': 'test_final',
                     'batch_size': 5}

# The data pipeline should be on CPU
with tf.device('/cpu:0'):
    CityData = dt.CityDataSet(test_data_params)
    next_batch = CityData.next_batch()

# Hparameter
model_params = {'num_classes': 19,
                'feed_weight': '../data/saved_weights/final_adam_batch3/watershed_presemgradswta1_final8s_ep3.npy',
                'batch_size': 5,
                'data_format': "NCHW", # optimal for cudnn
                }

num_val = 500
num_test = 1525
num_train = 2975
# iterations = 5
iterations = int(num_train / model_params['batch_size'])
batch = model_params['batch_size']

res38 = resnet38.ResNet38(model_params)
[sem_label, wt_label, sem_prob] = res38.inf(image=next_batch['img'])
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

    #print("Inference done! Start transforming to colored ...")
    #CityData.pred_to_color()
    print("Start transforming to labelIDs ...")
    CityData.pred_to_labelID()
    print("Transform done! Ready to generate evaluation files.")
    print("Start evaluating semantic accuracy ...")
    accuracy = evalPixelSemantic.run_eval('../data/sempred')
