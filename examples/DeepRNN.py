# -*- coding: utf-8 -*-
"""
@author: kstandvoss
"""

from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from hyperopt import hp, fmin, tpe

import numpy as np
import os
from matplotlib import pyplot as plt

class DeepRNN:
    """    
    :Description:
    """

    space = ( 
        hp.uniform( 'dropout', 0, 1 ),
        hp.choice( 'layer', [50, 100, 200]),
        hp.choice( 'activation', ['relu', 'tanh', 'sigmoid']),
        hp.choice( 'second', [False, True]),
        hp.choice( 'epochs', [1,3,5,10]),
        hp.choice( 'batch_size', [8,16,32,64,128])
    )

    def __init__(self,dropout, layer, activation, second, epochs, batch_size):
        """
        
        :Description:
            Initializes instance of DeepRNN.
        
        :Input parameters:
            input_dim:           Size of each input dimension (list)
            output_dim:          Size of each output dimension (list)
            
        """
        self.dropout = dropout
        self.layer = layer
        self.activation = activation
        self.second = second
        self.epochs = epochs
        self.batch_size = batch_size

        self.buildModel()
    
    def buildModel(self):

        self.model = Sequential()
        self.model.add(Embedding(256, 32, input_length=5))
        if self.second == 'two':
            self.model.add(GRU(self.layer, activation=self.activation, return_sequences=True, dropout_W=self.dropout, dropout_U=self.dropout))

        self.model.add(GRU(self.layer, activation=self.activation, return_sequences=False, dropout_W=self.dropout, dropout_U=self.dropout))    
        
        self.model.add(Dense(7))
        self.model.add(Activation('softmax'))            
        
    def train(self, X_train, Y_train, X_test, Y_test, noise=[0]):
        

        """
        
        :Description:
            Trains the network by going through the training data in random mini-batches for multiple epochs.
            
        :Input parameters:

            
        """

        """ compile """
        
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])  
        
        
        """ training procedure """    

        
        modelPath = './checkpoints/models/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        modelCheck = ModelCheckpoint(modelPath, monitor='val_loss', verbose=0,
                                        save_best_only=True, mode='auto')   
       

        SNRs = np.random.choice(noise,X_train.shape[0])
        noiseLVL = np.sqrt(np.var(X_train,1) / SNRs)
        noise = noiseLVL[...,None]*np.random.randn(X_train.shape[0],X_train.shape[1])
        X_train += noise

        history = self.model.fit(X_train, Y_train, 
                                    batch_size=self.batch_size, nb_epoch=self.epochs, shuffle=True,
                                    validation_data=(X_test,Y_test), callbacks= [modelCheck],verbose=2)

        score = self.model.evaluate(X_test,Y_test)

        return score[0]

    
    def evaluate(self, data, labels):
        
        scores = self.model.evaluate(data, labels, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
         
        return scores[1]*100    
