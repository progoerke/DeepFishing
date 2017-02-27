from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

class CCN():

    def __init__(self, size = (32, 32), nb_classes = 8, batch_size = 128, nb_filters = 32, pool_size = (2, 2), kernel_size = (3, 3), nb_epoch = 12):
        self.size = size
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.nb_filters = nb_filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.nb_epoch = nb_epoch


    def build(self):
        model = self.model = Sequential()

        model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1], border_mode='valid', input_shape=(3,self.size[0], self.size[1]), dim_ordering='th'))
        model.add(Activation('relu'))
        model.add(Convolution2D(self.nb_filters, self.kernel_size[0], self.kernel_size[1], dim_ordering='th'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = self.pool_size, dim_ordering='th'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        return model

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train, batch_size = self.batch_size, nb_epoch = self.nb_epoch, verbose = 1)

    def predict(self, X_test):
        self.model.predict_proba(X_test, batch_size = self.batch_size, verbose = 1)

