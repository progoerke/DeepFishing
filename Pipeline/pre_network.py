import os

from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import model_from_json

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from jsonloader import load_train

import tensorflow as tf

import numpy as np
import scipy.misc
from random import randint
import pickle 
import collections

import h5py
import sys

class Pre_Network:

    def __init__(self):
        #read data including img_height, img_width
        self.train_images, targets, self.masks, self.train_ids = load_train(use_cached=True,filepath='/work/kstandvoss/bb_train_mat.hdf5')
        self.test_images, self.test_ids = self.load_test()
        # 584, 565,
        # 31
        # 147, 142
        self.img_height = self.train_images.shape[1]
        self.img_width = self.train_images.shape[2]
        patch_size_height = 101
        patch_size_width = 101
        # Model
        base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (patch_size_height, patch_size_width, 3))
        x = base_model.output
        #x = ZeroPadding2D((1,1))(x)
        #x = Convolution2D(1024, 5,5, activation='relu')(x)
        #x = Convolution2D(1024, 4,4, activation='relu')(x)
        #x = Convolution2D(1024, 3,3, activation='relu')(x)
        x = Convolution2D(1024, 3,3, activation='relu')(x)
        predictions = Convolution2D(2, 1, 1, activation='relu')(x)
        #create graph of your new model
        self.new_model = Model(input = base_model.input, output = predictions)
        #compile the model
        self.new_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        print(self.new_model.summary())
    
    def load_test(self,filepath='/work/kstandvoss/test_mat.hdf5'):
        print('load from hdf5 file')
        file = h5py.File(filepath, "r")

        images = file["images"]
        ids = file['ids']

        sys.stdout.write('\n Doooone :)\n')
        return images, ids

    def shift_and_stitch(self,im, patch_size, stride):
        ''' Return a full resolution segmentation by segmenting shifted versions
        of the input 'im' and stitching those segmentations back together. 
        The stride determines how many times the image should be shifted in the x and y direction'''
            
        ## create the output shape, the output image is made a bit bigger to allow the stitched versions to be added. 
        ## the extra parts will be cut off at the end
        output = np.zeros([im.shape[0]+97, im.shape[0]+97])
        # the input image has to be padded with zeros to allow shifting of the image. 
        # pad input image (half filter size + stride)
        #im_padded = np.pad(im, ((patch_size//2, patch_size + stride),
        #                        (patch_size//2, patch_size + stride)), 'constant', constant_values = [0,0])    
        
        im_padded = np.zeros((im.shape[0]+97,im.shape[1]+97,im.shape[2]))

        im_padded[:,:,0] = np.pad(im[:,:,0], ((patch_size//2, patch_size + stride),
            (patch_size//2, patch_size + stride)), 'constant', constant_values = [0,0])
        im_padded[:,:,1] = np.pad(im[:,:,1], ((patch_size//2,patch_size + stride),
            (patch_size//2, patch_size + stride)), 'constant', constant_values = [0,0])
        im_padded[:,:,2] = np.pad(im[:,:,2], ((patch_size//2, patch_size + stride),
            (patch_size//2, patch_size + stride)), 'constant', constant_values = [0,0])

        im_p_sh = im_padded.shape
        
        # Now implement a loop that:
        # - obtains a shifted version of the image
        # - applies the fully convolutional network
        # - and places the network output in the output of this function
            
        for row in range(0, stride):
            for col in range(0, stride):
                sys.stdout.write(".")
                sys.stdout.flush()   
                        
                shifted_im = np.roll(im_padded, -row, axis=0)            
                shifted_im = np.roll(shifted_im, -col, axis=1)
                
                # forward pass
                probability = self.get99(shifted_im)
                for i in range(probability.shape[0]):
                    for j in range(probability.shape[1]):
                        output[row+i*stride,col+j*stride] = probability[i,j,1]
        
        return output[0:im.shape[0], 0:im.shape[1]]

    def get99(self,im,patch_size=49,stride=24):
        im_padded = np.zeros((im.shape[0]+patch_size-1,im.shape[1]+patch_size-1,im.shape[2]))
        im_padded[:,:,0] = np.pad(im[:,:,0], ((patch_size//2, patch_size//2),
                            (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
        im_padded[:,:,1] = np.pad(im[:,:,1], ((patch_size//2, patch_size//2),
                            (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
        im_padded[:,:,2] = np.pad(im[:,:,2], ((patch_size//2, patch_size//2),
                            (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
        output = np.zeros((16,16,2))
        for i in range(16):
            for j in range(16):
                x = np.expand_dims(im_padded[i*stride:i*stride+patch_size,j*stride:j*stride+patch_size,:], axis=0)
                output[i,j,:] = self.new_model.predict(x,verbose=0)
        
        return output
        

# AUCH AUF TEST train_images, ERSTELLE ZWEI HDF5 FILES

    def apply_shift_stitch(self,filepath_train='/work/kstandvoss/sas_train_mat.hdf5',filepath_test='/work/kstandvoss/sas_test_mat.hdf5'):

        print('create new train hdf5 file')
        file = h5py.File(filepath_train, "w")
        no_chunks = 2
        dt = h5py.special_dtype(vlen=bytes)

        #3777
        output = file.create_dataset("probas", (3777, self.img_height, self.img_width), chunks=(no_chunks, self.img_height, self.img_width), dtype='f', compression="lzf")
        ids = file.create_dataset("ids", (3777,1), chunks=(no_chunks,1), dtype=dt)
        total = 0
        for idx,img in enumerate(self.train_images):
            output[total,:,:] = self.shift_and_stitch(img,patch_size = 49,stride = 24)
            ids[total,:] = self.train_ids[idx]
            total += 1
            if idx%300==0:
                sys.stdout.write(".")
                sys.stdout.flush()
            if idx ==0:
                break
        file.flush()
        file.close()

        print('create new test hdf5 file')
        file = h5py.File(filepath_test, "w")
        no_chunks = 2
        dt = h5py.special_dtype(vlen=bytes)
        
        #1000
        output_test = file.create_dataset("probas", (1000, self.img_height, self.img_width), chunks=(no_chunks, self.img_height, self.img_width), dtype='f', compression="lzf")
        ids_test = file.create_dataset("ids", (1000,1), chunks=(no_chunks,1), dtype=dt)
        total = 0
        for idx,img in enumerate(self.test_images):
            output_test[total,:,:] = self.shift_and_stitch(img,49,49)
            ids_test[total,:] = self.test_ids[idx]
            total += 1
            if idx%100==0:
                sys.stdout.write(".")
                sys.stdout.flush()    
            if idx == 0:
                break
        file.flush()
        file.close()

    def load_model(self,model_name='/work/kstandvoss/new_model'):
        # load json and create model
        json_file = open(model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.new_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.new_model.load_weights(model_name + ".h5")
        print("Loaded model from disk")

    def extract_patches(self,imgs, msks,patch_size=101,stride=50):
        # if use_cached:
        #     print('load from patches hdf5 file')
        #     file = h5py.File(filepath_patches, "r")
        #     patches = file["patches"]
        #     labels = file["labels"]
        # else:
        #     num = self.train_images.shape[0]

        #     print('create new patches hdf5 file')
        #     file = h5py.File(filepath_patches, "w")
        #     no_chunks = 64
        #     dt = h5py.special_dtype(vlen=bytes)

        #     patches = file.create_dataset("patches", (num*5*5, patch_size, patch_size,3), chunks=(no_chunks, patch_size, patch_size,3), dtype='f', compression="lzf")
        #     labels = file.create_dataset("labels", (num*5*5,1,1,2), chunks=(no_chunks,1,1,2), dtype='i8')
            
        #     with tf.Session() as sess:
        #         size = 500
        #         for i in range(0,3777,size):
        #             print('patch {}:'.format(i))
        #             if i+size > 3777:
        #                 patch_arr = tf.extract_image_patches(self.train_images[i:,:,:,:], ksizes=[1, patch_size, patch_size, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME').eval()
        #                 dims = patch_arr.shape
        #                 patch_arr = np.reshape(patch_arr,(dims[0]*dims[1]*dims[2],patch_size,patch_size,3))
        #                 patches[i*dims[1]*dims[2]:] = patch_arr
        #             else:
        #                 patch_arr = tf.extract_image_patches(self.train_images[i:i+size,:,:,:], ksizes=[1, patch_size, patch_size, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME').eval()
        #                 dims = patch_arr.shape
        #                 patch_arr = np.reshape(patch_arr,(dims[0]*dims[1]*dims[2],patch_size,patch_size,3))
        #                 patches[i*dims[1]*dims[2]:(i+size)*dims[1]*dims[2]] = patch_arr

        #     sys.stdout.write('\n Created patches :)\n')
        #     sys.stdout.flush()

        #     for idx,p in enumerate(self.train_images.shape[0]):
        #         for i in range(dims[1]):
        #             for j in range(dims[2]):
        #                 #print(im_padded[i*stride:i*stride+patch_size,j*stride:j*stride+patch_size,:].shape)
        #                 #patches[total,:,:,:] = im_padded[i*stride:i*stride+patch_size,j*stride:j*stride+patch_size,:]
        #                 if self.masks[idx,i*stride,j*stride] == 1:
        #                     labels[total,:,:,1] = 1
        #                 else:
        #                     labels[total,:,:,0] = 1
        #                 total += 1
        #             if total%300==0:
        #                 sys.stdout.write(".")
        #                 sys.stdout.flush()


            #patches = np.zeros((self.train_images.shape[0]*9*9,49,49,3))
            #labels = np.zeros((self.train_images.shape[0]*9*9,1,1,2))
            # patches = np.zeros((num*16*16,49,49,3))
            # labels = np.zeros((num*16*16,1,1,2))
            # total = 0
            # for idx,im in enumerate(self.train_images):
            #     #im_padded = np.pad(im, ((49//2, 49 //2,0),(49//2, 49//2,0)), 'constant', constant_values = [0,0,0])
            #     im_padded = np.zeros((im.shape[0]+patch_size-1,im.shape[1]+patch_size-1,im.shape[2]))
            #     im_padded[:,:,0] = np.pad(im[:,:,0], ((patch_size//2, patch_size//2),
            #                         (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
            #     im_padded[:,:,1] = np.pad(im[:,:,1], ((patch_size//2, patch_size//2),
            #                         (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
            #     im_padded[:,:,2] = np.pad(im[:,:,2], ((patch_size//2, patch_size//2),
            #                         (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
            #     for i in range(16):
            #         for j in range(16):
            #             #print(im_padded[i*stride:i*stride+patch_size,j*stride:j*stride+patch_size,:].shape)
            #             patches[total,:,:,:] = im_padded[i*stride:i*stride+patch_size,j*stride:j*stride+patch_size,:]
            #             if self.masks[idx,i*stride,j*stride] == 1:
            #                 labels[total,:,:,1] = 1
            #             else:
            #                 labels[total,:,:,0] = 1
            #             total += 1
            #     if total%300==0:
            #         sys.stdout.write(".")
            # #         sys.stdout.flush()

            # file.flush()
            # file.close()
        with tf.Session() as sess:    
            patch_arr = tf.extract_image_patches(imgs, ksizes=[1, patch_size, patch_size, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME').eval()
            dims = patch_arr.shape
            patch_arr = np.reshape(patch_arr,(dims[0]*dims[1]*dims[2],patch_size,patch_size,3))

        labels = np.zeros((len(patch_arr),1,1,2))    
        total = 0
        for idx,p in enumerate(imgs):
            for i in range(dims[1]):
                for j in range(dims[2]):
                    #print(im_padded[i*stride:i*stride+patch_size,j*stride:j*stride+patch_size,:].shape)
                    #patches[total,:,:,:] = im_padded[i*stride:i*stride+patch_size,j*stride:j*stride+patch_size,:]
                    if np.sum(msks[idx,(i-1)*stride:(i+1)*stride,(j-1)*stride:(j+1)*stride]) >= 1000:
                        labels[total,:,:,1] = 1
                    else:
                        labels[total,:,:,0] = 1
                    total += 1
        # sys.stdout.write('\n Loaded :)\n')
        return patch_arr, labels    

    def train_model(self,model_name='/work/kstandvoss/new_model'):
        #train your model on data
        batch_size = 32
        for batch in range(0,len(self.train_images),batch_size):
            imgs = self.train_images[batch:batch+batch_size]
            msks = self.masks[batch:batch+batch_size]
            img_patches, patch_labels = self.extract_patches(imgs, msks)
            self.new_model.train_on_batch(img_patches, patch_labels)

        # serialize model to JSON
        model_json = self.new_model.to_json()
        with open(model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.new_model.save_weights(model_name + ".h5")
        print("Saved model to disk")

        #probability = shift_and_stitch(img_g, pad_size, 10)

#train_model()
