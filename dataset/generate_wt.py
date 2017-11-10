'''
    Give the following parameters:
        - CITYSCAPES_DATASET: './data/CityDatabase'(default)
    Output:
        - the corresponding ground truth discretized (K=[0,15]) distance transform for each '*_gt*_instanceIds.png' gt file
        - example:
            input file:  aachen_000000_000019_gtFine_instanceIds.png
            output file(as sparse matrix): aachen_000000_000019_gtFine_wt.png
        - this includes all object classes that have instance labels
        - instanceIds:
            person: 24000 - 24999
            rider: 25000 - 25999
            car: 26000 - 26999
            truck: 27000 - 27999
            bus: 28000 - 28999
            train: 31000 - 31999
            motorcycle: 32000 - 32999
            bicycle: 33000 - 33999
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
from scipy.misc import toimage
from scipy.ndimage.morphology import distance_transform_cdt
import multiprocessing

os.environ["CITYSCAPES_DATASET"] = "../data/CityDatabase"
# os.environ["CITYSCAPES_DATASET"] = "."
os.environ["NUM_PROCESSES"] = "25"

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
    # print('Got {} train_instance files.'.format(len(files_fine_train)))
    return files_fine

def open_gt_file(fname):
    ''' Open a single file given fname path'''

    img = Image.open(fname)
    image = np.array(img, dtype=np.int16)

    return image

def create_wt_per_image(image):
    ''' Given an numpy ndarray image, returns discretized distance transform:
        All pixel values have range of [0,15], integers'''
        # compute number of different instances: person,rider,car,truck,bus,train,motorcycle,bicycle
    ins_num = []
    instances = [24000,25000,26000,27000,28000,31000,32000,33000]
    for ins_class in instances:
        for i in range(ins_class,ins_class+999):
            if i in image:
                ins_num.append(i)
    # for each instance, generate a watershed_transform
    dist_trans_img = np.zeros((1024,2048),np.int32)
    for ins in ins_num:
        bool_mask = np.equal(image, ins)
        dist_trans = distance_transform_cdt(bool_mask)

        # discretize
        fk_start = 0
        for k in range(16):
            start = fk_start+k+1
            end = start+k+1
            np.place(dist_trans, np.logical_and(dist_trans>=start, dist_trans<=end), k)
            fk_start = start
        np.place(dist_trans, dist_trans>=136, 15)

        # stack together
        dist_trans_img += dist_trans

    # to color
    # wt_color = np.stack((dist_trans_img,dist_trans_img,dist_trans_img), axis=-1)
    # wt_color += np.ones((1024,2048,3), np.int32)*128

    # return (dist_trans_img, wt_color)
    return dist_trans_img

def main():
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapes_path = os.environ['CITYSCAPES_DATASET']
    files = get_file_list(cityscapes_path)

    if 'NUM_PROCESSES' in os.environ:
        num_processes = int(os.environ["NUM_PROCESSES"])
    else:
        num_processes = 6
    # progress = 0
    # print("Progress: {:>3} %".format( progress * 100 / len(files) ))

    process_pool = []
    chunk_size = int(len(files) / num_processes)

    for i in range(num_processes):
        if i != num_processes-1:
            process_pool.append(multiprocessing.Process(target=generate_wt, args=(files[i*chunk_size: (i+1)*chunk_size],)))
        else:
            process_pool.append(multiprocessing.Process(target=generate_wt, args=(files[i*chunk_size:],)))
    for i in range(num_processes):
        process_pool[i].start()
    for i in range(num_processes):
        process_pool[i].join()

    print("Generate wt done!")

def generate_wt(files):

    for fname in files:
        image = open_gt_file(fname)
        wt = create_wt_per_image(image)
        fname = fname.replace('instanceIds', 'wt')
        print('Save gt wt to {}.'.format(fname))
        toimage(wt, high=15, low=0, cmin=0, cmax=15, pal=None,mode=None).save(fname)

if __name__ == "__main__":
    main()


