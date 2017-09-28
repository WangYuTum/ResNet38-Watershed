from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
import data_utils as dt
from scipy.misc import imsave
from tensorflow.python import debug as tfdbg
from core import resnet38

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train_data_params = {'data_dir': '../data/CityDatabase',
                     'dataset': 'train_dir',
                     'batch_size': 1}
dataset = dt.CityDataSet(train_data_params)

model_params = {'num_classes': 19,
                'feed_weight': '../data/trained_weights/pretrained_ResNet38a1_imgnet.npy',
                'batch_size': 1,
                'decay_rate': 1e-5, ##as the paper
                'lr': 1e-5, ##NOTE, paper was 1e-5. Larger lr won't converge
                'save_path': '../data/saved_weights/',
                'tsboard_save_path': '../data/tsboard/'}

train_ep = 31
save_ep = 3
num_train = 2975

with tf.Session() as sess:
    #sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
    res38 = resnet38.ResNet38(model_params)
    save_path = model_params['save_path']
    batch_size = model_params['batch_size']

    train_img = tf.placeholder(tf.float32, shape=[batch_size, 1024, 2048, 3])
    train_sem_gt = tf.placeholder(tf.int32, shape=[batch_size, 1024, 2048])
    train_label = tf.placeholder(tf.float32, shape=[batch_size, 1024, 2048, 2])
    [train_op, loss] = res38.train_grad(image=train_img, sem_gt=train_sem_gt, label=train_label, params=model_params)

    save_dict_op = res38._var_dict
    TrainLoss_sum = tf.summary.scalar('train_loss', loss)
    Train_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(model_params['tsboard_save_path']+'grad_upsta2', sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    num_iters = np.int32(num_train / batch_size) + 1
    print('Start training...')
    for epoch in range(train_ep):
        print('Eopch %d'%epoch)
        for iters in range(num_iters):
            next_images, next_sem_gt, next_labels = dataset.next_batch() # images [batch_size,H,W,3], sem_gt [batch_size,H,W], labels [batch_size,H,W,2]
            train_feed_dict = {train_img: next_images, train_sem_gt: next_sem_gt, train_label: next_labels}
            [train_op_, loss_, Train_summary_] = sess.run([train_op, loss, Train_summary], train_feed_dict)
            writer.add_summary(Train_summary_, iters)
            if iters % 10 == 0:
                print('Iter {0} loss: {1}'.format(iters, loss_))
                # print(pred_.shape)
                # pred_img = np.concatenate((pred_, np.zeros((64,128,2),dtype=np.float32)), axis=-1)
                # s_img = np.array(pred_img[:,:,0:3])
                # imsave('pred_%d.png'%(iters), s_img)
        if epoch % save_ep == 0 and epoch !=0:
            print('Save trained weight after epoch: %d'%epoch)
            save_npy = sess.run(save_dict_op)
            save_path = model_params['save_path']
            if len(save_npy.keys()) != 0:
                save_name = 'watershed_preimgneta1_grad8s2_up_ep%d.npy'%(epoch)
                save_path = save_path + save_name
                np.save(save_path, save_npy)


