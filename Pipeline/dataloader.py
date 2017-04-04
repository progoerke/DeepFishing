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
import cv2

from heatmap_VisCAM import Heatmap
from imgloader import load_single_img

def load_test(use_cached=True,filepath='test_mat.hdf5',crop_rows=400,crop_cols=400,no=1000,use_heatmap=False):
    directories = "data/test_stg1"               #location of 'train'
    #subdirs = listdir(directories)[1::]
    #print(subdirs)

    num_total_images = no
    if use_cached is False:
        print('create new hdf5 file')
        file = h5py.File(filepath, "w")
        no_chunks = 2
        dt = h5py.special_dtype(vlen=bytes)
        
        images = file.create_dataset("images", (num_total_images, crop_rows, crop_cols, 3), chunks=(no_chunks, crop_rows, crop_cols, 3), dtype='f', compression="lzf")
        ids = file.create_dataset("ids", (num_total_images,1), chunks=(no_chunks,1), dtype=dt)
        crop_idx = file.create_dataset("crop_idx", (num_total_images,6), chunks=(no_chunks,1), dtype='int32')

        print('Read test images')
        total = 0
        files = listdir(directories) 
        h = Heatmap()
        for j, f in enumerate(files):           #parse through all files
            if ((j%100) == 0):
                sys.stdout.write(".")
                sys.stdout.flush()
            if not(f == '.DS_Store'):
                current_img = load_single_img(directories+"/"+f,convert_bgr=True)#img_rows, img_cols, color_type, interp=interp, img_as_float=img_as_float)

                #print(current_img.shape)

                if use_heatmap:
                    _,max_idx,_ = h.heatmap(current_img)
                    center_row = max_idx[0]
                    center_col = max_idx[1]
                    # Get from heatmap/box

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
                    crop_idx[total,0] = start_crop_row
                    crop_idx[total,1] = stop_crop_row
                    crop_idx[total,2] = start_crop_col
                    crop_idx[total,3] = stop_crop_col
                    crop_idx[total,4] = center_row
                    crop_idx[total,5] = center_col

                else:
                    current_img = current_img.astype('float32')
                    current_img /= 255
                    current_img = cv2.resize(current_img, (crop_cols, crop_rows))
                    crop_idx[total,:] = np.array([-1,-1,-1,-1,-1,-1])

                images[total, :, :, :] = current_img
                ids[total] = directories+"/"+f

                total += 1
        file.flush()

    else:
        print('load from hdf5 file')
        file = h5py.File(filepath, "r")

        images = file["images"]
        ids = file['ids']
        crop_idx = file['crop_idx']

    sys.stdout.write('\n Doooone :)\n')
    return images, ids, crop_idx

def load_train(use_cached=True,filepath='train_mat.hdf5',crop_rows=400,crop_cols=400,no=3777,use_heatmap=False):
    fish = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
    #fish = ['BET']
    directories = "data/train"               #location of 'train'
    #subdirs = listdir(directories)[1::]
    #print(subdirs)

    num_total_images = no
    if use_cached is False:
        print('create new hdf5 file')
        file = h5py.File(filepath, "w")
        #no_chunks = 64
        no_chunks = 2
        dt = h5py.special_dtype(vlen=bytes)
        images = file.create_dataset("images", (num_total_images, crop_rows, crop_cols, 3), chunks=(no_chunks, crop_rows, crop_cols, 3), dtype='f', compression="lzf")
        targets = file.create_dataset("targets", (num_total_images, 8), chunks=(no_chunks, 8), dtype='int32')
        ids = file.create_dataset("ids", (num_total_images,1), chunks=(no_chunks,1), dtype=dt)
        crop_idx = file.create_dataset("crop_idx", (num_total_images,6), chunks=(no_chunks,1), dtype='int32')

        print('Read train images')
        total = 0
        h = Heatmap()
        for i,d in enumerate(fish): #parse all subdirections
            sys.stdout.write(".")
            sys.stdout.flush()
            
            files = listdir(directories+"/"+d)  
            for j, f in enumerate(files):           #parse through all files
            #print(f)
                if not(f == '.DS_Store'):
                    current_img = load_single_img(directories+"/"+d+"/"+f,convert_bgr=True)
                    #print(directories+"/"+d+"/"+f)
                    #print(current_img.shape)

                    if use_heatmap:
                        _,max_idx,_ = h.heatmap(current_img)
                        print(max_idx)
                        center_row = max_idx[0]
                        center_col = max_idx[1]
                    # Get from heatmap/box

                        start_crop_row = int(center_row - crop_rows/2)
                        if start_crop_row < 0:
                            start_crop_row = 0
                        stop_crop_row = int(start_crop_row + crop_rows)
                        if stop_crop_row > current_img.shape[0]:
                            stop_crop_row = current_img.shape[0]
                            start_crop_row = stop_crop_row - crop_rows
                        start_crop_col = int(center_col - crop_cols/2)
                        if start_crop_col < 0:
                            start_crop_col = 0
                        stop_crop_col = int(start_crop_col + crop_cols)
                        if stop_crop_col > current_img.shape[1]:
                            stop_crop_col = current_img.shape[1]
                            start_crop_col = stop_crop_col - crop_cols

                        crop_idx[total,0] = start_crop_row
                        crop_idx[total,1] = stop_crop_row
                        crop_idx[total,2] = start_crop_col
                        crop_idx[total,3] = stop_crop_col
                        crop_idx[total,4] = center_row
                        crop_idx[total,5] = center_col

                        current_img = current_img.astype('float32')
                        current_img /= 255

                        current_img = current_img[start_crop_row:stop_crop_row,start_crop_col:stop_crop_col,:]

                    else:
                        current_img = current_img.astype('float32')
                        current_img /= 255
                        current_img = cv2.resize(current_img, (crop_cols, crop_rows))
                        crop_idx[total,:] = np.array([-1,-1,-1,-1,-1,-1])

                    images[total, :, :, :] = current_img
                    targets[total, :] = 0
                    targets[total, i] = 1
                    ids[total] = directories+"/"+d+"/"+f

                    total += 1
        file.flush()

    else:
        print('load from hdf5 file')
        file = h5py.File(filepath, "r")

        images = file["images"]
        targets = file["targets"]
        ids = file["ids"]
        crop_idx = file['crop_idx']

    sys.stdout.write('\n Doooone :)\n')
    return images, targets, ids, crop_idx

#load_train(filepath='just_test.hdf5',use_cached=False, use_heatmap = False,no=80)
#load_test(filepath='/work/kstandvoss/test_mat.hdf5',use_cached=False, use_heatmap = False)
#load_max_idx()
#load_train(use_cached=False)

# start = time.time()
# load_test(use_cached=False,crop_rows=200,crop_cols=200)
# end = time.time()
# print(end - start)
##626.100456237793
##548.030868053
#start = time.time()
#load_train(use_cached=False,filepath='train_mat_smaller.hdf5',crop_rows=400,crop_cols=400, use_heatmap=True)
#end = time.time()
#print(end - start)
#plt.imshow([[1,0],[0,1]])
#plt.show()
