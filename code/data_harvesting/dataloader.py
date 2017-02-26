from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
from  os import listdir
import pickle
import os
import glob
import math
import time

from scipy.misc import imread, imresize

def load(use_chached=True,filepath='data_mat.hdf5',crop_rows=200,crop_cols=200,no=3777):
    fish = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
    directories = "../../data/train"               #location of 'train'
    #subdirs = listdir(directories)[1::]
    #print(subdirs)

    num_total_images = no
    if create_new is True:
        print('create new hdf5 file')
        file = h5py.File(filepath, "w")

        images = file.create_dataset("images", (num_total_images, 3, crop_rows, crop_cols), chunks=(64, 3, crop_rows, crop_cols), dtype='f', compression="lzf")
        targets = file.create_dataset("targets", (num_total_images, 8), chunks=(64, 8), dtype='int32')

        print('Read train images')
        total = 0
        for i,d in enumerate(fish): #parse all subdirections
            sys.stdout.write(".")
            sys.stdout.flush()
            
            files = listdir(directories+"/"+d)  
            for j, f in enumerate(files):           #parse through all files
            #print(f)
                if not(f == '.DS_Store'):
                    current_img = imread(directories+"/"+d+"/"+f)#img_rows, img_cols, color_type, interp=interp, img_as_float=img_as_float)
                    current_img = current_img[:, :, ::-1] # convert to bgr
                    current_img = current_img.transpose((2, 0, 1))
                    current_img = current_img.astype('float32')
                    current_img /= 255

                    #print(current_img.shape)

                    # Get from heatmap/box
                    center_row = 250
                    center_col = 500
                    start_crop_row = int(center_row - crop_rows/2)
                    if start_crop_row < 0:
                        start_crop_row = 0
                    stop_crop_row = int(start_crop_row + crop_rows)
                    if stop_crop_row > current_img.shape[1]:
                        stop_crop_row = current_img.shape[1]
                        start_crop_row = stop_crop_row - crop_rows
                    start_crop_col = int(center_row - crop_cols/2)
                    if start_crop_col < 0:
                        start_crop_col = 0
                    stop_crop_col = int(start_crop_col + crop_cols)
                    if stop_crop_col > current_img.shape[2]:
                        stop_crop_col = current_img.shape[2]
                        start_crop_col = stop_crop_col - crop_cols

                    current_img = current_img[:,start_crop_row:stop_crop_row,start_crop_col:stop_crop_col]
                    images[total, :, :, :] = current_img
                    targets[total, :] = 0
                    targets[total, i] = 1

                    total += 1
        file.flush()

    else:
        print('load from hdf5 file')
        file = h5py.File(filepath, "r")

        images = file["images"]
        targets = file["targets"]

    sys.stdout.write('\n Doooone :)\n')
    return images, targets

#626.100456237793
start = time.time()
load(create_new=True,crop_rows=200,crop_cols=200)
end = time.time()
print(end - start)