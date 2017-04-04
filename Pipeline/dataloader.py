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
from skimage import io
from blurBoats_v2 import boat_clusters
from blurBoats_v2 import average_clusters
from blurBoats_v2 import blurBoat

from heatmap_VisCAM import Heatmap
from imgloader import load_single_img
from keras.preprocessing.image import img_to_array
import json

def load_test(use_cached=True,filepath='test_mat.hdf5',directories = 'data/test_stg1', crop_rows=400,crop_cols=400,no=1000,mode="resize"):
    #directories = "data/test_stg1"

    #hm_directories = directories.replace('/test/', '/test/hm_') # heatmap directory
    #if not os.path.isdir(hm_directories):
    #    os.makedirs(hm_directories)
    #crop_directories = directories.replace('/test/', '/test/cr_') # cropped images directory
    #if not os.path.isdir(crop_directories):
    #   os.makedirs(crop_directories)

    #location of 'train'
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

                if mode is "use_heatmap_sliding":
                    stride = 112
                    best_prob = 0
                    n0 = int(current_img.shape[0] / stride)
                    n1 = int(current_img.shape[1] / stride)
                    for i0 in range(n0):
                        for i1 in range(n1):
                            sliding_img = current_img[i0 * stride:min(current_img.shape[0], (i0 + 2) * stride),
                                          i1 * stride:min(current_img.shape[1], (i1 + 2) * stride)]
                            probs = h.model.predict(np.array([img_to_array(cv2.resize(sliding_img, (224, 224)))]))
                            if sum(probs[0, h.pred_class]) >= best_prob:
                                best_prob = sum(probs[0, h.pred_class])
                                best_img = sliding_img
                                best_i0 = i0
                                best_i1 = i1
                    _, heatmap_overlay, best_max_idx, prob = h.heatmap(best_img)
                    #io.imsave(directories.replace('/test/', '/test/hm_') + "/" + f, heatmap_overlay)
                    max_idx = [best_i0 * stride + best_max_idx[0], best_i1 * stride + best_max_idx[1]]
                    print("heatmap max_idx: ", max_idx)
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

                    current_img = current_img[start_crop_row:stop_crop_row,start_crop_col:stop_crop_col,:]
                    crop_idx[total,0] = start_crop_row
                    crop_idx[total,1] = stop_crop_row
                    crop_idx[total,2] = start_crop_col
                    crop_idx[total,3] = stop_crop_col
                    crop_idx[total,4] = center_row
                    crop_idx[total,5] = center_col

                elif mode is "use_heatmap":
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

                elif mode is "resize":
                    current_img = current_img.astype('float32')
                    current_img /= 255
                    current_img = cv2.resize(current_img, (crop_cols, crop_rows))
                    crop_idx[total,:] = np.array([-1,-1,-1,-1,-1,-1])

                #io.imsave(directories.replace('/test/', '/test/cr_') + "/" + f, current_img)
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

def load_train(use_cached=True,filepath='train_mat.hdf5',crop_rows=400,crop_cols=400,no=3777,mode="resize"):
    fish = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
    #fish = ['ALB','DOL','LAG']
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

        # Loading the boat cluster labels and average images
        ncluster = 22
        imgs_averages = [None] * ncluster
        y = np.loadtxt('./data/train/Boat_clusters_553/img_labels_y.txt')
        f = open('./data/train/Boat_clusters_553/img_file_names.json', 'r')
        all_file_names = json.load(f)
        f.close
        y = y.astype(int)
        for i in range(len(imgs_averages)):
            imgs_averages[i] = io.imread('./data/train/Boat_clusters_553/imgs_averages_' + str(i) + '.jpg')
        y_file_names=[y, all_file_names]
        cluster_size = np.zeros((np.max(y) + 1, 1), dtype=int)
        for icluster in range(np.max(y) + 1):
            cluster_size[icluster] = np.sum(y == icluster)

        print('Read train images')
        total = 0
        h = Heatmap()
        for i,d in enumerate(fish): #parse all subdirections
            sys.stdout.write(".")
            sys.stdout.flush()

            directories_fish=directories + "/" + d
            hm_directories_fish = directories_fish.replace('/train/', '/train/hm_')  # heatmap directory
            if not os.path.isdir(hm_directories_fish):
                os.makedirs(hm_directories_fish)
            crop_directories_fish = directories_fish.replace('/train/', '/train/cr_')  # cropped images directory
            if not os.path.isdir(crop_directories_fish):
                os.makedirs(crop_directories_fish)

            files = listdir(directories+"/"+d)  
            for j, f in enumerate(files):           #parse through all files
                print("Fish #", i+1, ": ", fish[i], ", image # ", j+1, ": ", f)
                if not(f == '.DS_Store'):
                    current_img = load_single_img(directories+"/"+d+"/"+f,convert_bgr=True)

                    # Blurring the boat
                    icluster = y_file_names[0][y_file_names[1].index("./"+directories+"/"+d+"\\"+f)]
                    if cluster_size[icluster] > 3 and icluster>0:
                        current_img = blurBoat(icluster, current_img, imgs_averages, maxblur=1.0)
                    #else:
                    #    current_img = img

                    if mode is "use_heatmap_sliding":
                        stride=112
                        best_prob=0
                        n0 = int(current_img.shape[0] / stride)
                        n1 = int(current_img.shape[1] / stride)
                        for i0 in range(n0):
                            for i1 in range(n1):
                                sliding_img=current_img[i0*stride:min(current_img.shape[0], (i0+2)*stride), i1*stride:min(current_img.shape[1],(i1+2)*stride)]
                                probs = h.model.predict(np.array([img_to_array(cv2.resize(sliding_img,(224,224)))]))
                                if sum(probs[0,h.pred_class])>=best_prob:
                                    best_prob=sum(probs[0,h.pred_class])
                                    best_img=sliding_img
                                    best_i0=i0
                                    best_i1=i1
                        _, heatmap_overlay, best_max_idx, prob = h.heatmap(best_img)
                        io.imsave(directories+"/hm_"+d+"/"+f, heatmap_overlay)
                        max_idx=[best_i0*stride+best_max_idx[0], best_i1*stride+best_max_idx[1]]
                        #print("heatmap max_idx: ", max_idx)
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

                    elif mode is "use_heatmap":
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

                    elif mode is "resize":
                        current_img = current_img.astype('float32')
                        current_img /= 255
                        current_img = cv2.resize(current_img, (crop_cols, crop_rows))
                        crop_idx[total,:] = np.array([-1,-1,-1,-1,-1,-1])

                    io.imsave(directories+"/cr_"+d+"/"+f, current_img)
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
