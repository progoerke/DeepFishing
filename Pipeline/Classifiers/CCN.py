from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping

from hyperopt import hp

class CCN():

    space = (
        hp.choice('batch_size', [8,16,32,64,128]),
        hp.choice('nb_filters', [16, 32, 64]),
        hp.choice('pool_size', [(2,2), (3,3), (4,4)]),
        hp.choice('kernel_size', [(3,3), (4,4), (5,5)]),
        hp.choice('optimizer', ['adam', 'adadelta'])
    )

    def __init__(self, size = (32, 32), nb_classes = 8, nb_epoch = 12, batch_size = 128, nb_filters = 32, pool_size = (2, 2), kernel_size = (3, 3), optimizer='adadelta'):
        self.size = size
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.nb_filters = nb_filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        
        self.model = self.build()

    def build(self):
        model = self.model = Sequential()

        model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1], border_mode='valid', input_shape=(self.size[0], self.size[1],3), dim_ordering='tf'))
        model.add(Activation('relu'))
        model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1], dim_ordering='tf'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = self.pool_size, dim_ordering='tf'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        return model

    def fit(self, X_train, Y_train, X_val, Y_val):

        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

        self.model.fit(X_train, Y_train, validation_data=(X_val, Y_val), callbacks=[early], batch_size = self.batch_size, nb_epoch = self.nb_epoch, verbose = 1)

    def predict(self, X_test):
        return self.model.predict_proba(X_test, batch_size = self.batch_size, verbose = 1)

    def evaluate(self, X_val, Y_val):
        
        score = self.model.evaluate(X_val, Y_val)    
        return score[0]

