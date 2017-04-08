import dataloader
import pickle
import numpy as np
from Classifiers.inceptionV3 import Inception
from Classifiers.resNet import ResNet
from Classifiers.VGG_16 import VGG_16
from Classifiers.CCN import CCN

import hyper as HY

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

import VGG_CV as CV
from keras.optimizers import SGD
from hyper import train_generator
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from keras.layers.core import Dense, Flatten
from keras.layers import AveragePooling2D
from keras.models import Model
from keras.applications import InceptionV3
from keras import optimizers
from keras.layers.core import Dense, Activation, Merge

###################################################################################
###################################################################################

class Stacking(object):
    def __init__(self, n_folds, stacker_args, base_models_args):
        self.n_folds = n_folds
        self.stacker_args = stacker_args
        self.base_models_args = base_models_args[:n_folds]
        self.n_classes = 8

    def fit_predict(self, data, labels, cval_indx ,test):

        s_train = np.zeros((data.shape[0], len(self.base_models_args)*self.n_classes))
        s_test = np.zeros((test.shape[0], len(self.base_models_args)*self.n_classes))
        print(len(self.base_models_args))
        for i in range(len(self.base_models_args)):
            print("Training model {}/{}".format(i, len(self.base_models_args)))
            s_test_i = np.zeros((self.n_folds, test.shape[0], 8))
            for j in range(self.n_folds):
                print("Training without fold ̣{}".format(j))
                # Divide training and validation
                indx = [np.where(cval_indx == ind) for ind in np.unique(cval_indx)]
                selector = [x for k,x in enumerate(indx) if k != j][0]
                train_indx = selector[0].tolist()
                train_indx.sort()
                selector = [x for k, x in enumerate(indx) if k == j][0]
                val_indx = selector[0].tolist()
                val_indx.sort()
                # Load the model
                print("Loading model...")
                base_model = self.load_base_model(self.base_models_args[i])
                # Train
                print("Training model...")
                model = self.train_base_model(classifier_name = self.base_models_args[i][0], classifier=base_model,
                                         val_fold=j, data=data, labels=labels, train_indx=train_indx, val_indx=val_indx)
                print("Model saved")
                # Get predictions
                val_set = [data[k] for k in val_indx]
                y_pred = base_model.predict(val_set)
                s_train[val_indx, i*self.n_classes:i*self.n_classes+self.n_classes] = y_pred
                s_test_i[j,:,:] = model.predict(test)

            # Average test predictions
            tmp = np.mean( np.array(s_test_i), axis=0 )
            s_test[:, i*self.n_classes:i*self.n_classes+8] = tmp

        # Save train and test set for the stacker
        s_train = np.savetxt('data/train_stacker.csv')
        s_test = np.savetxt('data/test_stacker.csv')

        # Create stacker
        print("Training stacker model...")
        # OPTION 1 Neural net
        stacker = self.load_stacker_s()
        stacker = self.train_stacker_s(stacker, s_train, labels)
        # OPTION 2 Simple logistic regression
        # stacker = LogisticRegression(multi_class='multinomial', class_weight='balanced')
        # cross_val_score(stacker,s_train[:-1],s_train[-1],cv=10)
        # stacker.fit(s_train[:-1],s_train[-1])

        print("Making the final predictions...")
        y_pred = stacker.predict(s_test)

        # OPTION 3
        stacker = self.load_stacker_c(data[0].shape)
        stacker = self.train_stacker_c(stacker, data, s_train, labels)

        return y_pred

    def load_base_model(self, args):
        name = args.pop(0)
        if name == 'inception':
            return Inception((data[0].shape[0], data[0].shape[1]), 8, 50, *args)
        elif name == 'resnet':
            return ResNet((data[0].shape[0], data[0].shape[1]), 8, 50, *args)
        elif name == 'vgg':
            return VGG_16((data[0].shape[0], data[0].shape[1]), 8, 50, *args)
        elif name == 'ccn':
            return CCN((data[0].shape[0], data[0].shape[1]), 8, 50, *args)


    def train_base_model(self, classifier_name=None, classifier=None, val_fold=None, data=None, labels=None, train_indx=None, val_indx=None):
        # Instances generator
        tg = train_generator(data, labels, train_indx, classifier.batch_size)
        vg = train_generator(data, labels, val_indx, classifier.batch_size)

        classifier.fit(tg, vg, (len(train_indx), len(val_indx)))

        # Save classifier
        model = classifier.model
        model.save_weights('/models/stacking/{}_val{}.h5'.format(classifier_name, val_fold))
        model_json = model.to_json()
        with open('/models/stacking/{}_val{}.json'.format(classifier_name, val_fold), "w") as json_file:
            json_file.write(model_json)

        return classifier

    def load_stacker_s(self):
        model = Sequential()
        model.add(Dense(1024, input_shape=(len(self.base_models_args) * 8,)))
        model.add(Dense(8, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], validation_split=0.2)
        print(model.summary())
        return model

    def load_stacker_c(self, input_size):

        # Image model
        inception = InceptionV3(include_top=False, weights='imagenet', input_shape=input_size)

        for layer in inception.layers:
            layer.trainable = False

        output = inception.output
        output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
        output = Flatten(name='flatten')(output)
        image_processor = Model(inception.input, output)

        # Metadata model
        metadata_processor = Sequential()
        metadata_processor.add(Dense(1024, input_shape=(len(self.base_models_args) * 8,)))
        metadata_processor.add(Dense(8, activation='softmax'))

        # Concatenate both

        model = Sequential()
        model.add(Merge([image_processor, metadata_processor], merge_mode='concat'))
        model.add(Dense(8096, input_dim = image_processor.output_shape[1] + 8))  #check for this one
        model.add(Activation('relu'))
        model.add(Dense(self.n_classes, activation='softmax', name='predictions'))

        opt = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=True)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=["accuracy"])
        return model

    def train_stacker_s(self, model, s_train, labels):
        # define the checkpoint
#        filepath = 'models/stacker_model_loss-{loss:.4f}.hdf5'
#        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
#        callbacks_list = [checkpoint]
        # Early stopping
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        # Fit the model
        model.fit(s_train, labels, nb_epoch=100, batch_size=32, callbacks=early, shuffle='batch')

        # Save model
        model.save_weights('models/resnet.h5')
        # serialize model to JSON
        model_json = model.to_json()
        with open('model_json/resnet.json', "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

        return model

    def train_stacker_c(self, model, data, s_train, labels):
        # define the checkpoint
 #       filepath = 'models/stacker_model_loss-{loss:.4f}.hdf5'
 #       checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True)
 #       callbacks_list = [checkpoint]
        # Early stopping
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        # Fit the model
        model.fit([data, s_train], labels, nb_epoch=100, batch_size=32, callbacks=early, shuffle='batch')

        # Save model
        model.save_weights('models/stacker.h5')
        # serialize model to JSON
        model_json = model.to_json()
        with open('model_json/resnet.json', "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")
        return model

###################################################################################
###################################################################################

if __name__ == '__main__':
    models_dir = "models/"
    model_arg_files = ['resnet_loss-1.4480618794978652.pkl',
                       'resnet_loss-1.9494444401883289.pkl']
    stacker_arg_file = []
    n_folds = 2
    use_cached = True
    # Read parameters from file
    print("Reading parameters for the models...")
    model_arg_list = []
    stacker_arg = []
    for file_name in model_arg_files:
        param = pickle.load(open(file_name, "rb"))
        name = [file_name.split('_')[0]]
        model_arg_list.append(name + param)
#    stacker_arg = pickle.load(open(stacker_arg_file, "rb"))
    print("Parameters read")
    # Prepare data
    print("Loading training data...")
    data, labels, _, _ = dataloader.load_train(filepath='data/train.hdf5',use_cached=use_cached)
    print("Training data loaded")
    print("Creating {}-folds...".format(n_folds))
    cval_indx = CV.VGG_CV(data, labels, folds=n_folds, use_cached = True, path_cached='data/cv_data.pkl')
    print("Folds created")
    print("Load test data")
    test, filenames, _ = dataloader.load_test(filepath='data/test_stg1.hdf5', use_cached=use_cached)
    print("Test data loaded")
    # Run everything
    print("Running stacking...")
    st = Stacking(n_folds, stacker_arg, model_arg_list)
    predictions = st.fit_predict(data, labels, cval_indx, test)
    print("Saving results...")
    HY.write_submission('stacking_submission', predictions, filenames)
    print("Stacking finished")
