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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 9.0

test_data_params = {'data_dir': '../data/CityDatabase',
                     'dataset': 'val',
                     'batch_size': 1,
                     'pred_save_path': '../data/pred_trainIDs',
                     'colored_save_path': '../data/pred_colored',
                     'labelIDs_save_path': '../data/pred_labelIDs'}

dataset = dt.CityDataSet(test_data_params)

model_params = {'num_classes': 19,
                'feed_path': 'data/trained_weights/???'}
num_val = 1525
num_test = 500
iterations = 1

# with tf.Session() as sess:
with tf.Session(config=config_gpu) as sess:
    res38 = resnet38.ResNet38(model_params)

    # Feed test/val image, batch=1
    img = tf.placeholder(tf.float32, shape=[1, 1024, 2048, 3])
    # Get inference result
    predict = res38.inf(image)

    print('Finished building inference network ResNet38-8s')
    accuracy = 0.0
    init = tf.global_variables_initializer()
    sess.run(init)

    print('Start inference...')
    for i in range(iterations):
        print('iter {0}:'.format(i))
        next_pair = dataset.next_batch()
        next_pair_image = next_pair[0]
        feed_dict = {image: next_pair_image}

        pred_out = sess.run(predict, feed_dict=feed_dict)
        dataset.save_trainID_img(pred_out)

    print("Inference done! Start transforming to colored ...")
    dataset.pred_to_color()
    print("Start transforming to labelIDs ...")
    dataset.pred_to_labelID()
    print("Start evaluating accuracy ...")
    accuracy = evalPixelSemantic.run_eval(test_data_params['labelIDs_save_path'])
    print("Final score {}".format(accuracy))

