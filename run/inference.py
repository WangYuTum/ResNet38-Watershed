from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
import data_utils as dt
from core import resnet38
from eval import evalPixelSemantic

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.9

test_data_params = {'data_dir': '../data/CityDatabase',
                     'dataset': 'val',
                     'batch_size': 1,
                     'pred_save_path': '../data/pred_trainIDs',
                     'colored_save_path': '../data/pred_colored',
                     'labelIDs_save_path': '../data/pred_labelIDs'}

dataset = dt.CityDataSet(test_data_params)

model_params = {'num_classes': 19,
                'feed_weight': '../data/saved_weights/watershed_precitya1_sem8s_momen_up_ep5.npy'}
num_val = 500
num_test = 1525
iterations = 4

with tf.Session() as sess:
#with tf.Session(config=config_gpu) as sess:
    res38 = resnet38.ResNet38(model_params)

    # Feed test/val image, batch=1
    img = tf.placeholder(tf.float32, shape=[1, 1024, 2048, 3])
    # Get inference result
    predict = res38.inf(img)

    print('Finished building inference network ResNet38-8s')
    accuracy = 0.0
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
        dataset.save_trainID_img(pred_out)

    print("Inference done! Start transforming to colored ...")
    dataset.pred_to_color()
    # print("Start transforming to labelIDs ...")
    # dataset.pred_to_labelID()
    # print("Start evaluating accuracy ...")
    # accuracy = evalPixelSemantic.run_eval(test_data_params['labelIDs_save_path'])
    # print("Final score {}".format(accuracy))

