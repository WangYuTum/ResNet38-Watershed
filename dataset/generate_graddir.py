'''
    Give the following parameters:
        - CITYSCAPES_DATASET: './data/CityDatabase'(default)
    Output:
        - the corresponding ground truth gradient direction of distance transform for each '*_gt*_instanceIds.png' gt file
        - example:
            input file:  aachen_000000_000019_gtFine_instanceIds.png
            output file(as sparse matrix): aachen_000000_000019_gtFine_graddir.npz
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import sys
import os
import glob
from scipy.misc import imsave
from scipy.ndimage import distance_transform_edt
from scipy.sparse import save_npz
from scipy.sparse import csc_matrix

os.environ["CITYSCAPES_DATASET"] = "./data/CityDatabase"
# os.environ["CITYSCAPES_DATASET"] = "/Users/WY/Desktop/TUM_profile/MyGit/ResNet38-Watershed/dataset"

def get_file_list(cityscapes_path):
    ''' Get all .png files of intance'''

    search_fine_train = os.path.join( cityscapes_path , "gtFine" , "train" , "*" , "*_gt*_instanceIds.png")
    search_fine_val = os.path.join( cityscapes_path , "gtFine" , "val" , "*" , "*_gt*_instanceIds.png")
    # search_fine_train = os.path.join( cityscapes_path , "*_gt*_instanceIds.png")

    files_fine_train = glob.glob(search_fine_train)
    files_fine_val = glob.glob(search_fine_val)
    files_fine = files_fine_train + files_fine_val
    # files_fine = files_fine_train
    files_fine.sort()

    if not files_fine:
        sys.exit('Did not find any files!')
    print('Got {} train_instance files, {} val_instance files.'.format(len(files_fine_train), len(files_fine_val)))
    return files_fine

def open_gt_file(fname):
    ''' Open a single file given fname path'''

    img = Image.open(fname)
    image = np.array(img, dtype=np.int16)

    return image

def create_graddir_per_image(image):
    ''' Given an numpy ndarray image, returns a colored grad image:
        The 1st channel is the x gradient, the 2nd channel is the y gradient,
        the 3rd channle if filled with zeros. The gradient vectors of each pixel are normalized to have a unit length'''
    # compute number of cars
    ins_num = []
    for i in range(26000,26999):
        if i in image:
            ins_num.append(i)
    # for each instance, generate a graddir
    grad_colors = np.zeros((1024,2048,3))
    for i in ins_num:
        bool_mask = np.equal(image, i)
        dist_trans = distance_transform_edt(bool_mask)
        [x_grad, y_grad] = np.gradient(dist_trans)
        # normalize the gradients
        norm_matrix = np.sqrt(x_grad*x_grad+y_grad*y_grad)
        zero_mask = np.equal(norm_matrix,0)
        np.place(norm_matrix,zero_mask,1)
        x_grad_normed = x_grad / norm_matrix
        y_grad_normed = y_grad / norm_matrix
        # norm done
        grad_color = np.stack((x_grad,y_grad,np.zeros((1024,2048))), axis=-1)
        grad_colors += grad_color

    # grad_colors.astype(np.float16)
    return grad_colors

def main():
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapes_path = os.environ['CITYSCAPES_DATASET']
    files = get_file_list(cityscapes_path)

    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ))
    for fname in files:
        image = open_gt_file(fname)
        graddir = create_graddir_per_image(image)
        fname = fname.replace('instanceIds', 'graddir')
        fname = fname.replace('.png','.npz')
        print('Save gt graddir to {}.'.format(fname))
        graddir = np.reshape(graddir, (1024,2048*3))
        graddir_sparse = csc_matrix(graddir)
        save_npz(fname, graddir_sparse)

        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ))
        sys.stdout.flush()

    print("Generate graddir done!")

if __name__ == "__main__":
    main()








