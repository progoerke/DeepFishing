import json
import h5py
import sys
import os
import numpy as np
import cv2
from imgloader import load_single_img

def load_train(use_cached=True,filepath='bb_train_mat.hdf5',crop_rows = 400,crop_cols = 400 ,no=3777):
    fish = ['ALB','BET','DOL','LAG','OTHER','SHARK','YFT']
    #fish = ['SHARK']
    img_directories = "data/train"               #location of 'train'
    bb_directories = "bb_data"
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
        masks = file.create_dataset("masks", (num_total_images, crop_rows, crop_cols), chunks = (no_chunks, crop_rows, crop_cols), dtype = 'f', compression="lzf")
        ids = file.create_dataset("ids", (num_total_images,1), chunks=(no_chunks,1), dtype=dt)

        print('Read train images')
        total = 0
        for i,d in enumerate(fish): #parse all subdirections
            sys.stdout.write(".")
            sys.stdout.flush()
            

            with open(bb_directories+'/'+d.lower()+'_labels.json') as json_data:
                json_data = json.load(json_data)
                for x in json_data:
                    #load_single_img('data/train/ALB/img_00003.jpg',convert_bgr=True)
                    #print(img_directories+"/"+d+"/")
                    #print(x['filename'])
                    #print(img_directories+"/"+d+"/"+x['filename'])
                    current_img = load_single_img(img_directories+"/"+d+"/"+x['filename'],convert_bgr=True)
                    current_msk = np.zeros((current_img.shape[0], current_img.shape[1]))
                    for bb in x['annotations']:
                    	# x,y is top left corner of bb
                        # cleaning:
                        y_val = int(bb['y'])
                        x_val = int(bb['x'])        
                        height_val = int(bb['height'])
                        width_val = int(bb['width'])
                        
                        #start_row = current_msk.shape[0]-y_val
                        start_row = y_val
                        if start_row < 0:
                            start_row = 0

                        #stop_row = current_msk.shape[0]-y_val+height_val
                        stop_row = y_val+height_val
                        if stop_row > current_msk.shape[0]:
                            stop_row = current_msk.shape[0]-1

                        start_col = x_val
                        if x_val < 0:
                            x_val = 0

                        stop_col = x_val+width_val
                        if stop_col > current_msk.shape[1]:
                            stop_col = current_msk.shape[1]-1
                    
                        current_msk[start_row:stop_row, start_col:stop_col] = 1

                    current_msk = current_msk.astype('float32')
                    current_msk = cv2.resize(current_msk, (crop_cols, crop_rows))

                    current_img = current_img.astype('float32')
                    current_img /= 255
                    current_img = cv2.resize(current_img, (crop_cols, crop_rows))

                    images[total, :, :, :] = current_img
                    targets[total, :] = 0
                    targets[total, i] = 1
                    masks[total,:,:] = current_msk
                    ids[total] = img_directories+"/"+d+"/"+x['filename']

                    total += 1
            file.flush()

    else:
        print('load from hdf5 file')
        file = h5py.File(filepath, "r")

        images = file["images"]
        targets = file["targets"]
        masks = file["masks"]
        ids = file["ids"]

    sys.stdout.write('\n Doooone :)\n')
    return images, targets, masks, ids