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

from heatmap_VisCAM import heatmap
from imgloader import load_single_img

from scipy.misc import imread, imresize

def load_test(use_chached=True,filepath='test_mat.hdf5',crop_rows=200,crop_cols=200,no=1000,use_heatmap=False):
    directories = "../../data/test_stg1"               #location of 'train'
    #subdirs = listdir(directories)[1::]
    #print(subdirs)

    num_total_images = no
    if use_chached is False:
        print('create new hdf5 file')
        file = h5py.File(filepath, "w")

        images = file.create_dataset("images", (num_total_images, crop_rows, crop_cols, 3), chunks=(64, crop_rows, crop_cols, 3), dtype='f', compression="lzf")

        print('Read test images')
        total = 0
        files = listdir(directories) 
        for j, f in enumerate(files):           #parse through all files
            if ((j%100) == 0):
                sys.stdout.write(".")
                sys.stdout.flush()
            if not(f == '.DS_Store'):
                current_img = load_single_img(directories+"/"+f)#img_rows, img_cols, color_type, interp=interp, img_as_float=img_as_float)

                #print(current_img.shape)

                if use_heatmap:
                    _,max_idx,_ = heatmap(current_img)
                    center_row = max_idx[0]
                    center_col = max_idx[1]
                # Get from heatmap/box
                else:
                    center_row = 250
                    center_col = 500


                start_crop_row = int(center_row - crop_rows/2)
                if start_crop_row < 0:
                    start_crop_row = 0
                stop_crop_row = int(start_crop_row + crop_rows)
                if stop_crop_row > current_img.shape[0]:
                    stop_crop_row = current_img.shape[0]
                    start_crop_row = stop_crop_row - crop_rows
                start_crop_col = int(center_row - crop_cols/2)
                if start_crop_col < 0:
                    start_crop_col = 0
                stop_crop_col = int(start_crop_col + crop_cols)
                if stop_crop_col > current_img.shape[1]:
                    stop_crop_col = current_img.shape[1]
                    start_crop_col = stop_crop_col - crop_cols

                current_img = current_img[start_crop_row:stop_crop_row,start_crop_col:stop_crop_col,:]
                images[total, :, :, :] = current_img

                total += 1
        file.flush()

    else:
        print('load from hdf5 file')
        file = h5py.File(filepath, "r")

        images = file["images"]

    sys.stdout.write('\n Doooone :)\n')
    return images

def load_train(use_chached=True,filepath='train_mat.hdf5',crop_rows=200,crop_cols=200,no=3777,use_heatmap=False):
    fish = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
    directories = "../../data/train"               #location of 'train'
    #subdirs = listdir(directories)[1::]
    #print(subdirs)

    num_total_images = no
    if use_chached is False:
        print('create new hdf5 file')
        file = h5py.File(filepath, "w")

        images = file.create_dataset("images", (num_total_images, crop_rows, crop_cols, 3), chunks=(64, crop_rows, crop_cols, 3), dtype='f', compression="lzf")
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
                    current_img = load_single_img(directories+"/"+f)
                    
                    #print(current_img.shape)

                    if use_heatmap:
                        _,max_idx,_ = heatmap(current_img)
                        center_row = max_idx[0]
                        center_col = max_idx[1]
                    # Get from heatmap/box
                    else:
                        center_row = 250
                        center_col = 500

                    start_crop_row = int(center_row - crop_rows/2)
                    if start_crop_row < 0:
                        start_crop_row = 0
                    stop_crop_row = int(start_crop_row + crop_rows)
                    if stop_crop_row > current_img.shape[0]:
                        stop_crop_row = current_img.shape[0]
                        start_crop_row = stop_crop_row - crop_rows
                    start_crop_col = int(center_row - crop_cols/2)
                    if start_crop_col < 0:
                        start_crop_col = 0
                    stop_crop_col = int(start_crop_col + crop_cols)
                    if stop_crop_col > current_img.shape[1]:
                        stop_crop_col = current_img.shape[1]
                        start_crop_col = stop_crop_col - crop_cols

                    current_img = current_img[start_crop_row:stop_crop_row,start_crop_col:stop_crop_col,:]
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

# start = time.time()
# load_test(use_chached=False,crop_rows=200,crop_cols=200)
# end = time.time()
# print(end - start)
##626.100456237793
##548.030868053
start = time.time()
load_train(use_chached=False,crop_rows=200,crop_cols=200)
end = time.time()
print(end - start)