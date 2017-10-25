from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
import data_utils as dt
from core import resnet38

# Prepare dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train_data_params = {'mode': 'train_sem',
                     'batch_size': 3}
# The data pipeline should be on CPU
with tf.device('/cpu:0'):
    CityData = dt.CityDataSet(train_data_params)
    next_batch = CityData.next_batch()

# Hparameter
model_params = {'num_classes': 19,
                'feed_weight': '../data/saved_weights/sem2_adam_batch3_stage2/watershed_precitya1_8s_ep15.npy',
                'batch_size': 3,
                'decay_rate': 0.0005,
                'lr': 0.0008,
                'data_format': "NCHW", # optimal for cudnn
                'save_path': '../data/saved_weights/',
                'tsboard_save_path': '../data/tsboard/'}
train_ep = 46
save_ep = 5
num_train = 2975

# Build network
# This part should be on GPU
res38 = resnet38.ResNet38(model_params)
[train_op, loss] = res38.train_sem(image=next_batch['img'], label=next_batch['sem_gt'], params=model_params)
save_dict_op = res38._var_dict
TrainLoss_sum = tf.summary.scalar('train_loss', loss)
Train_summary = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    save_path = model_params['save_path']
    batch_size = model_params['batch_size']
    writer = tf.summary.FileWriter(model_params['tsboard_save_path']+'sem2/adam_batch3_stage2/', sess.graph)

    sess.run(init)
    num_iters = np.int32(num_train / batch_size) + 1
    print('Start training...')
    for epoch in range(train_ep):
        print('Eopch %d'%epoch)
        for iters in range(num_iters):
            [train_op_, loss_, Train_summary_] = sess.run([train_op, loss, Train_summary])
            writer.add_summary(Train_summary_, iters)
            if iters % 10 == 0:
                print('Iter {} loss: {}'.format(iters, loss_))
        if epoch % save_ep == 0 and epoch !=0:
            print('Save trained weight after epoch: %d'%epoch)
            save_npy = sess.run(save_dict_op)
            save_path = model_params['save_path']
            if len(save_npy.keys()) != 0:
                save_name = '/sem2_adam_batch3_stage2/watershed_precitya1_8s_ep%d.npy'%(epoch+15)
                save_path = save_path + save_name
                np.save(save_path, save_npy)


