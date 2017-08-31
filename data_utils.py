from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from dataset.CityDataSet import CityDataSet


def load_weight(path):

    if path is None:
        sys.exit('Load weight is None!')
    fpath = os.path.abspath(os.path.join(path, os.curdir))
    data_dict = np.load(fpath, encoding='latin1').item()
    print('Loaded weights file %s'%fpath)
    return data_dict


