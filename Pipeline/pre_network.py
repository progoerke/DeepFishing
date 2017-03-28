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

import numpy as np
from random import randint
import pickle 
import collections

import h5py
import sys

class Pre_Network:

    def __init__(self):
        #read data including img_height, img_width
        self.train_images, targets, self.masks, self.train_ids = load_train(use_cached=True)
        self.test_images, self.test_ids = self.load_test()
        # 584, 565,
        # 31
        # 147, 142
        self.img_height = self.train_images.shape[1]
        self.img_width = self.train_images.shape[2]
        patch_size_height = 49
        patch_size_width = 49
        # Model
        base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (patch_size_height, patch_size_width, 3))
        x = base_model.output
        #x = ZeroPadding2D((1,1))(x)
        #x = Convolution2D(1024, 5,5, activation='relu')(x)
        #x = Convolution2D(1024, 4,4, activation='relu')(x)
        #x = Convolution2D(1024, 3,3, activation='relu')(x)
        #x = Convolution2D(1024, 3,3, activation='relu')(x)
        predictions = Convolution2D(2, 1, 1, activation='relu')(x)
        #create graph of your new model
        self.new_model = Model(input = base_model.input, output = predictions)
        #compile the model
        self.new_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        #print(self.new_model.summary())
    
    def load_test(self,filepath='test_mat.hdf5'):
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
        output = np.zeros([im.shape[0]+122, im.shape[0]+122])
        # the input image has to be padded with zeros to allow shifting of the image. 
        # pad input image (half filter size + stride)
        #im_padded = np.pad(im, ((patch_size//2, patch_size + stride),
        #                        (patch_size//2, patch_size + stride)), 'constant', constant_values = [0,0])    
        
        im_padded = np.zeros((im.shape[0]+122,im.shape[1]+122,im.shape[2]))

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

    def get99(self,full_img,patch_size=49):
        im_padded = np.zeros((full_img.shape[0]+48,full_img.shape[1]+48,full_img.shape[2]))
        im_padded[:,:,0] = np.pad(full_img[:,:,0], ((patch_size//2, patch_size//2),
                            (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
        im_padded[:,:,1] = np.pad(full_img[:,:,1], ((patch_size//2, patch_size//2),
                            (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
        im_padded[:,:,2] = np.pad(full_img[:,:,2], ((patch_size//2, patch_size//2),
                            (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
        output = np.zeros((9,9,2))
        for i in range(9):
            for j in range(9):
                x = np.expand_dims(im_padded[i*49:i*49+49,j*49:j*49+49,:], axis=0)
                output[i,j,:] = self.new_model.predict(x,verbose=0)
        
        return output
        

# AUCH AUF TEST train_images, ERSTELLE ZWEI HDF5 FILES

    def apply_shift_stitch(self,filepath_train='sas_train_mat.hdf5',filepath_test='sas_test_mat.hdf5'):

        print('create new train hdf5 file')
        file = h5py.File(filepath_train, "w")
        no_chunks = 2
        dt = h5py.special_dtype(vlen=bytes)

        #3777
        output = file.create_dataset("probas", (5, self.img_height, self.img_width, 3), chunks=(no_chunks, self.img_height, self.img_width, 3), dtype='f', compression="lzf")
        ids = file.create_dataset("ids", (5,1), chunks=(no_chunks,1), dtype=dt)
        total = 0
        for idx,img in enumerate(self.train_images):
            output[total,:,:,:] = self.shift_and_stitch(img,49,49)
            ids[total,:] = self.ids[idx]
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
        output_test = file.create_dataset("probas", (5, img_height, img_width, 3), chunks=(no_chunks, img_height, img_width, 3), dtype='f', compression="lzf")
        ids_test = file.create_dataset("ids", (5,1), chunks=(no_chunks,1), dtype=dt)
        total = 0
        for idx,img in enumerate(self.test_images):
            output_test[total,:,:,:] = self.shift_and_stitch(img,49,49)
            ids_test[total,:] = self.ids[idx]
            total += 1
            if idx%100==0:
                sys.stdout.write(".")
                sys.stdout.flush()    
            if idx == 0:
                break
        file.flush()
        file.close()

    def load_model(self,model_name='new_model'):
        # load json and create model
        json_file = open(model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.new_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.new_model.load_weights(model_name + ".h5")
        print("Loaded model from disk")

    def extract_patches(self,patch_size=49,stride=49):
        #patches = np.zeros((self.train_images.shape[0]*9*9,49,49,3))
        #labels = np.zeros((self.train_images.shape[0]*9*9,1,1,2))
        patches = np.zeros((3*9*9,49,49,3))
        labels = np.zeros((3*9*9,1,1,2))
        total = 0
        for idx,im in enumerate(self.train_images):
            #im_padded = np.pad(im, ((49//2, 49 //2,0),(49//2, 49//2,0)), 'constant', constant_values = [0,0,0])
            im_padded = np.zeros((im.shape[0]+48,im.shape[1]+48,im.shape[2]))
            im_padded[:,:,0] = np.pad(im[:,:,0], ((patch_size//2, patch_size//2),
                                (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
            im_padded[:,:,1] = np.pad(im[:,:,1], ((patch_size//2, patch_size//2),
                                (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
            im_padded[:,:,2] = np.pad(im[:,:,2], ((patch_size//2, patch_size//2),
                                (patch_size//2, patch_size//2)), 'constant', constant_values = [0,0])
            for i in range(9):
                for j in range(9):
                    patches[total,:,:,:] = im_padded[i*49:i*49+49,j*49:j*49+49,:]
                    if self.masks[idx,i*49,j*49] == 1:
                        labels[total,:,:,1] = 1
                    else:
                        labels[total,:,:,0] = 1
                    total += 1
            if total%300==0:
                sys.stdout.write(".")
                sys.stdout.flush()
            if idx == 2:
                break
        return patches, labels    

    def train_model(self,model_name='new_model'):
        #train your model on data
        img_patches, patch_labels = self.extract_patches()
        self.new_model.fit(img_patches, patch_labels, batch_size = 32, verbose = 1)

        # serialize model to JSON
        model_json = self.new_model.to_json()
        with open(model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.new_model.save_weights(model_name + ".h5")
        print("Saved model to disk")

        #probability = shift_and_stitch(img_g, pad_size, 10)

#train_model()