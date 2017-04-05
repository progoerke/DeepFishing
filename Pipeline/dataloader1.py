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
'''from blurBoats_v2 import boat_clusters
from blurBoats_v2 import average_clusters
from blurBoats_v2 import blurBoat'''

from heatmap_VisCAM import Heatmap
from imgloader import load_single_img
from fish_utils import crop_around
from keras.preprocessing.image import img_to_array
import json

def load_test(use_cached=True,filepath='test_mat.hdf5',directories = 'data/test_stg1', crop_rows=400,crop_cols=400,no=1000,mode="resize",rotate = False):
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
                    p = heatmap_slider(current_img,h)

                elif mode is "use_heatmap":
                    _,max_idx,_ = h.heatmap(current_img)
                    center_row = max_idx[0]
                    center_col = max_idx[1]
                    # Get from heatmap/box

                    crop_around(current_img,crop_cols,crop_rows,center_col,center_row)


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

def load_train(use_cached=True,filepath='train_mat.hdf5',crop_rows=400,crop_cols=400,no=3777,mode="resize",rotate = False, blur = False):
    fish = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
    #fish = ['ALB','DOL','LAG']
    directories = "F:/mlip/train1"               #location of 'train'
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
        if blur:
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
                    if blur:
                        icluster = y_file_names[0][y_file_names[1].index("./"+directories+"/"+d+"\\"+f)]
                        if cluster_size[icluster] > 3 and icluster>0:
                            current_img = blurBoat(icluster, current_img, imgs_averages, maxblur=1.0)
                            
                    if mode is "use_heatmap_sliding":
                        P=heatmap_slider(current_img, h)
                        P=P+np.min(P)
                        P=P/np.max(P)
                        
                        pts,vals,arr = fish_locator_3000(P,current_img.shape[1],current_img.shape[0])
                        for i,p in enumerate(pts):
                            img1 = zoom_around(current_img,p[1],p[0],1.2/np.exp(vals[i]))
                            crpd = crop_around(img1,crop_cols,crop_rows,p[0],p[1])[0]
                            io.imsave(directories+"/cr_"+d+"/"+str(p)+f, crpd)
                            #cv2.imshow(str(p),crpd)
                            if i>5: break

                        
                    elif mode is "use_heatmap":
                        _,max_idx,_ = h.heatmap(current_img)
                        center_row = max_idx[0]
                        center_col = max_idx[1]
                        # Get from heatmap/box
                        crop_around(current_img,crop_cols,crop_rows,center_col,center_row)
                        
                    elif mode is "resize":
                        current_img = current_img.astype('float32')
                        current_img /= 255
                        current_img = cv2.resize(current_img, (crop_cols, crop_rows))
                        crop_idx[total,:] = np.array([-1,-1,-1,-1,-1,-1])

                    #io.imsave(directories+"/cr_"+d+"/"+f, current_img)
                    #images[total, :, :, :] = current_img
                    targets[total, :] = 0
                    targets[total, i] = 1
                    #ids[total] = directories+"/"+d+"/"+f

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


def heatmap_slider(img, h, stride = 112):
    '''Slides a heatmap detector of size stride across img
    
    
    '''
    n0 = int(img.shape[0] / stride)
    n1 = int(img.shape[1] / stride)
    
    P = np.zeros((n0,n1))
    P_img = np.zeros(img.shape)
    
    best_prob=0
    for i0 in range(n0):
        for i1 in range(n1):
            #get window coords:
            y_ = i0*stride      
            _y = min(img.shape[0], (i0+2)*stride)
            x_ = i1*stride
            _x = min(img.shape[1],(i1+2)*stride)
            
            sliding_img=img[y_:_y, x_:_x]
            
            probs = h.model.predict(np.array([img_to_array(cv2.resize(sliding_img,(224,224)))]))
            P[i0,i1] = sum(probs[0,h.pred_class])      #Matrix of p

    return P
    '''if sum(probs[0,h.pred_class])>=best_prob:
                best_prob=sum(probs[0,h.pred_class])
                best_img=sliding_img
                best_i0=i0
                best_i1=i1'''
    
                
    '''_, heatmap_overlay, best_max_idx, prob = h.heatmap(best_img)
    io.imsave(directories+"/hm_"+d+"/"+f, heatmap_overlay)
    max_idx=[best_i0*stride+best_max_idx[0], best_i1*stride+best_max_idx[1]]
    #print("heatmap max_idx: ", max_idx)
    center_row = max_idx[0]
    center_col = max_idx[1]'''
    # Get from heatmap/box



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
