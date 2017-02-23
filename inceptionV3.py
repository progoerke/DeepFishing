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
from keras import optimizers

###################################### ----- INCEPTION V3 MODEL ----- #################################################

def inception_preprocess(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


class Inception():
    """
    The InceptionV3 Imagenet model with Batch Normalization for the Dense Layers
    """
    def __init__(self, size=(224, 224), n_classes=2, lr=0.001, batch_size=64):
        self.size = size
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size

    def build(self):
        """
        Loads preconstructed inception model from keras without top classification layer;
        Stacks custom classification layer on top;
        Returns stacked model
        """
        img_input = Input(shape=(3, self.size[0], self.size[1]))
        inception = InceptionV3(include_top=False, weights='imagenet', input_tensor=img_input)

        for layer in inception.layers:
            layer.trainable = False

        output = inception.output
        output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        output = Dense(self.n_classes, activation='softmax', name='predictions')(output)

        model = self.model = Model(inception.input, output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=True),
                      metrics=["accuracy"])
        return model

    def get_datagen(self, aug=False):
        if aug:
            return ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                      channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                      horizontal_flip=True, preprocessing_function=inception_preprocess)
        return ImageDataGenerator(preprocessing_function=inception_preprocess)

    def fit_val(self, trn_path, val_path, nb_trn_samples, nb_val_samples, nb_epoch=1, callbacks=[], aug=False):
        """
        Custom fit method for training with validation data and option for data augmentation"
        """
        train_datagen = self.get_datagen(aug=aug)
        val_datagen = self.get_datagen(aug=False)

        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        val_gen = val_datagen.flow_from_directory(val_path, target_size=self.size, batch_size=self.batch_size,
                                                  class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                                 validation_data=val_gen, nb_val_samples=nb_val_samples, callbacks=callbacks)


    def fit_full(self, trn_path, nb_trn_samples, nb_epoch=1, callbacks=[], aug=False):
        """
        Custom fit method for training without validation data and option for data augmentation
        """
        train_datagen = self.get_datagen(aug=aug)

        trn_gen = train_datagen.flow_from_directory(trn_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode='categorical', shuffle=True)
        self.model.fit_generator(trn_gen, samples_per_epoch=nb_trn_samples, nb_epoch=nb_epoch, verbose=2,
                                 callbacks=callbacks)

    def test(self, test_path, nb_test_samples, aug=False):
        """
        Custom prediction method with option for data augmentation
        """
        test_datagen = self.get_datagen(aug=aug)
        test_gen = test_datagen.flow_from_directory(test_path, target_size=self.size, batch_size=self.batch_size,
                                                    class_mode=None, shuffle=False)
        return self.model.predict_generator(test_gen, val_samples=nb_test_samples), test_gen.filenames
