from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from dataset.CIFAR10 import CIFAR10


def load_weight(path):

    fpath = os.path.abspath(os.path.join(path, os.curdir))
    data_dict = np.load(fpath, encoding='latin1').item()
    print('Loaded weights file %s'%fpath)
    return data_dict


