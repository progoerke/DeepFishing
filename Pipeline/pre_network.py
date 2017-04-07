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
        self.train_images, targets, self.masks, self.train_ids = load_train(use_cached=True,filepath='data/bb_train_mat.hdf5')
        
        self.img_height = self.train_images.shape[1]
        self.img_width = self.train_images.shape[2]
        patch_size_height = 101
        patch_size_width = 101
        # Model
        base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (patch_size_height, patch_size_width, 3))

        for layer in base_model.layers:
          layer.trainable = False

        x = base_model.output
        x = Convolution2D(1024, 3,3, activation='relu')(x)
        predictions = Convolution2D(2, 1, 1, activation='relu')(x)
        #create graph of your new model
        self.new_model = Model(input = base_model.input, output = predictions)
        #compile the model
        self.new_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        print(self.new_model.summary())
    
    def load_test(self,filepath='data/test_mat.hdf5'):
        print('load from hdf5 file')
        file = h5py.File(filepath, "r")

        images = file["images"]
        ids = file['ids']

        sys.stdout.write('\n Doooone :)\n')
        return images, ids

    def load_model(self,model_name='new_model'):
        # load json and create model
        json_file = open(model_name + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.new_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.new_model.load_weights(model_name + ".h5")
        print("Loaded model from disk")

    def extract_patches(self,imgs, msks=None, patch_size=101,stride=50):
        with tf.Graph().as_default(), tf.Session() as sess:  
          patch_arr = tf.extract_image_patches(imgs, ksizes=[1, patch_size, patch_size, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='SAME').eval()
          dims = patch_arr.shape
          patch_arr = np.reshape(patch_arr,(dims[0]*dims[1]*dims[2],patch_size,patch_size,3))

        labels = np.zeros((len(patch_arr),1,1,2))    
        if msks is not None: 
          total = 0
          for idx,p in enumerate(imgs):
              for i in range(dims[1]):
                  for j in range(dims[2]):
                      if np.sum(msks[idx,(i-1)*stride:(i+1)*stride,(j-1)*stride:(j+1)*stride]) >= 2000:
                          labels[total,:,:,1] = 1
                      else:
                          labels[total,:,:,0] = 1
                      total += 1
        return patch_arr, labels    

    def train_model(self,model_name='new_model', epochs=1):
        #train your model on data
        batch_size = 8
        for epoch in range(epochs):
          for batch in range(0,len(self.train_images),batch_size):
              imgs = self.train_images[batch:batch+batch_size]
              msks = self.masks[batch:batch+batch_size]
              img_patches, patch_labels = self.extract_patches(imgs, msks)
              weights = np.ones(len(img_patches))
              weights[np.where(patch_labels[...,1] == 1)[0]] = 30
              loss = self.new_model.train_on_batch(img_patches, patch_labels, sample_weight=weights)
              print('epoch {}: loss = {}'.format(epoch,loss))

        # serialize model to JSON
        model_json = self.new_model.to_json()
        with open(model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.new_model.save_weights(model_name + ".h5")
        print("Saved model to disk")

    def predict(self,X_test):
      img_patches,_ = self.extract_patches(X_test)
      prediction = self.new_model.predict_on_batch(img_patches)
      pos_patches = img_patches[np.argmax(prediction.squeeze(),axis=1) == 1,:,:,:]  
      if len(pos_patches)== 0:
        pos_patches = img_patches[np.argmax(prediction.squeeze(),axis=1) == 0,:,:,:]  
      
      return pos_patches
