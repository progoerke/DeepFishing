### credit rdcolema

import numpy as np

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D, MaxPooling2D, GlobalAveragePooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Activation, Lambda
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.applications.resnet50 import identity_block, conv_block
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.callbacks import EarlyStopping
from keras import optimizers


from Classifiers.classifier_base import Classifier_base

from math import log

from hyperopt import hp
###################################### ----- INCEPTION V3 MODEL ----- #################################################

def inception_preprocess(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


class Inception(Classifier_base):


    space = (
        hp.loguniform('lr', log(1e-4), log(1)),
        hp.choice('batch_size',[8,16,32,64]),
        hp.choice('optimizer',['adam','adadelta','sgd'])
    )
    """
    The InceptionV3 Imagenet model with Batch Normalization for the Dense Layers
    """
    def __init__(self, size=(224, 224), n_classes=2, nb_epoch = 12, lr=0.001, batch_size=64, optimizer='adam'):
        self.size = size
        self.n_classes = n_classes
        self.nb_epoch = nb_epoch
        self.lr = lr
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.class_weight = None
        self.model = self.build()

    def build(self):
        """
        Loads preconstructed inception model from keras without top classification layer;
        Stacks custom classification layer on top;
        Returns stacked model
        """
        #img_input = Input(shape=(self.size[0], self.size[1], 3))
        inception = InceptionV3(include_top=False, weights='imagenet', input_shape=(self.size[0], self.size[1], 3))

        for layer in inception.layers:
            layer.trainable = False

        output = inception.output
        output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(self.n_classes, activation='softmax', name='predictions')(output)

        model = self.model = Model(inception.input, output)

        if self.optimizer == 'adam':
            opt = optimizers.adam(lr=self.lr)
        elif self.optimizer == 'adadelta':
            opt = optimizers.adadelta(lr = self.lr)
        else:
            opt=optimizers.SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=True)
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=["accuracy"])
        return model

    def fit(self, train_generator, validation_generator, split):

        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

        self.model.fit_generator(train_generator, steps_per_epoch=split[0]//self.batch_size, validation_data=validation_generator, validation_steps=split[1]//self.batch_size, callbacks=[early], epochs = self.nb_epoch, verbose = 1)


    def predict(self, X_test):
        return self.model.predict(X_test, batch_size = self.batch_size, verbose = 1)

    def evaluate(self, validation_generator, split):
        
        score = self.model.evaluate_generator(validation_generator, steps=split//self.batch_size)    
        return score[0] 
