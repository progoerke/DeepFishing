import dataloader
import pickle
import numpy as np
from Classifiers.inceptionV3 import Inception
from Classifiers.resNet import ResNet
from Classifiers.VGG_16 import VGG_16
from Classifiers.CCN import CCN

import hyper as HY

from keras.applications import ResNet50
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

from keras.layers.core import Dense
from keras.layers import Input, concatenate

###################################################################################
###################################################################################

class Stacking(object):

    def __init__(self, n_folds, stacker_args, base_models_args, n_classes):
        self.n_folds = n_folds
        # Check that the number of folds correspond to the number of base_models
        if n_folds != len(base_models_args):
            raise Warning("The number of folds does no correspond to the number of base models.\n"
                          "Using just the first {} base models".format(n_folds))
        self.stacker_args = stacker_args
        self.base_models_args = base_models_args[:n_folds]
        self.n_classes = n_classes

    def fit_predict(self, data, labels, cval_indx, test):

        s_train = np.zeros((data.shape[0], self.n_classes))
        s_test = np.zeros((test.shape[0], self.n_classes))
        print(len(self.base_models_args))
        for i in range(len(self.base_models_args)):
            s_test_tmp = np.zeros((self.n_folds, test.shape[0], 8))
            print("Training model {}/{}".format(i+1, len(self.base_models_args)))
            # Divide training and validation
            indx = [np.where(cval_indx == ind) for ind in np.unique(cval_indx)]
            selector = [x for k,x in enumerate(indx) if k != i][0]
            train_indx = selector[0].tolist()
            train_indx.sort()
            selector = [x for k, x in enumerate(indx) if k == i][0]
            val_indx = selector[0].tolist()
            val_indx.sort()

            # Load the model
            print("Loading model...")
            base_model = self.load_model(self.base_models_args[i], data[0].shape[:2])
            # Train
            print("Training model...")
            model = self.train_base_model(classifier_name = self.base_models_args[i][0], classifier=base_model,
                                     val_fold=i, data=data, labels=labels, train_indx=train_indx, val_indx=val_indx)
            print("Model saved")
            # Get predictions
            val_set = np.array([data[k] for k in val_indx])
            y_pred = base_model.predict(val_set)
            s_train[val_indx,:] = y_pred
            s_test_tmp[i,:,:] = model.predict(test)

        # Average test predictions
        s_test = np.mean( np.array(s_test_tmp), axis=0 )

        # Save train and test set for the stacker
        np.savetxt('data/train_stacker.csv',s_train)
        np.savetxt('data/test_stacker.csv',s_test)

        # Create stacker
        print("Training stacker model...")
        # OPTION 1 Neural net
        # stacker = self.load_stacker_s()
        # stacker = self.train_stacker_s(stacker, s_train, labels)
        # OPTION 2 Simple logistic regression
        # stacker = LogisticRegression(multi_class='multinomial', class_weight='balanced')
        # cross_val_score(stacker,s_train[:-1],s_train[-1],cv=10)
        # stacker.fit(s_train[:-1],s_train[-1])
        # OPTION 3 Convolutional net with metadata
        stacker = self.load_stacker_c(self.stacker_args, data[0].shape[:2])
        stacker = self.train_stacker_c(stacker, data, s_train, labels)

        print("Making the final predictions...")
        y_pred = stacker.predict(s_test)

        return y_pred

    def load_model(self, args, input_size):
        name = args.pop(0)
        if name == 'inception':
            return Inception(input_size, 8, 50, *args)
        elif name == 'resnet':
            return ResNet(input_size, 8, 50, *args)
        elif name == 'vgg':
            return VGG_16(input_size, 8, 50, *args)
        elif name == 'ccn':
            return CCN(input_size, 8, 50, *args)
        return []


    def train_base_model(self, classifier_name=None, classifier=None, val_fold=None, data=None, labels=None, train_indx=None, val_indx=None):
        # Instances generator
        tg = train_generator(data, labels, train_indx, classifier.batch_size)
        vg = train_generator(data, labels, val_indx, classifier.batch_size)

        classifier.fit(tg, vg, (len(train_indx), len(val_indx)))

        # Save classifier
        model = classifier.model
        model.save_weights('models/stacking/{}_val{}.h5'.format(classifier_name, val_fold))
        model_json = model.to_json()
        with open('models/stacking/{}_val{}.json'.format(classifier_name, val_fold), "w") as json_file:
            json_file.write(model_json)

        return classifier

    def load_stacker_s(self):
        model = Sequential()
        model.add(Dense(1024, input_shape=(len(self.base_models_args) * 8,)))
        model.add(Dense(8, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], validation_split=0.2)
        print(model.summary())
        return model

    def load_stacker_c(self, stacker_args, input_size):

        # Image model
        model_name = stacker_args.pop(0)
        lr = stacker_args.pop(0)
        optimizer = stacker_args[-1]

        if model_name == 'inception':
            conv_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_size)
        else:
            conv_model = ResNet50(include_top=False,weights='imagenet', input_shape=input_size)

        for layer in conv_model.layers:
            layer.trainable = False

        conv_model_extra = conv_model.output
        conv_model_extra = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(conv_model_extra)
        conv_model_extra = Flatten(name='flatten')(conv_model_extra)

        # image_processor = Model(inception.input, output)

        # Metadata model
        meta_input = Input(shape=(len(self.base_models_args) * 8,))
        meta_dense = Dense(1024)(meta_input)
        meta_output = Dense(8, activation='softmax')(meta_dense)

        # Concatenate both

        merge_one = concatenate([conv_model_extra, meta_output])

        merge_dense1 = Dense(8096)(merge_one)
        merge_dense2 = Dense(8, activation='softmax', name='predictions')(merge_dense1)

        model = Model(inputs=[conv_model.input, meta_input], outputs=merge_dense2)

        if optimizer == 'adam':
            opt = optimizers.adam(lr=lr)
        elif optimizer == 'adadelta':
            opt = optimizers.adadelta(lr = lr)
        else:
            opt=optimizers.SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=True)

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
        model.fit([data, s_train], labels, nb_epoch=50, batch_size=64, callbacks=early, shuffle='batch')

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
    model_arg_files = ['inception_fine_loss-1.6862200698411534.pkl',
                       'inception_fine_loss-1.7430047196459908.pkl',
                       'resnet_loss-2.091301735867275.pkl',
                       'resnet_norm-144806.pkl']
    stacker_arg_file = ['inception_fine_loss-2.929521924498453.pkl']
    n_folds = 4
    n_classes = 8
    use_cached = True
    # Read parameters from file
    print("Reading parameters for the models...")
    model_arg_list = []
    stacker_arg = []
    for file_name in model_arg_files:
        a = models_dir + file_name
        param = list(pickle.load(open(models_dir + file_name, "rb")))
        name = [file_name.split('_')[0]]
        model_arg_list.append(name + param)
    if stacker_arg_file:
        param = pickle.load(open(models_dir + stacker_arg_file[0], "rb"))
        name = [stacker_arg_file[0].split('_')[0]]
        stacker_arg = name + param
    print("Parameters read")
    # Prepare data
    print("Loading training data...")
    data, labels, _, _ = dataloader.load_train(filepath='data/train_heat.hdf5', use_cached=use_cached)
    print("Training data loaded")
    print("Creating {}-folds...".format(n_folds))
    cval_indx = CV.VGG_CV(data, labels, folds=n_folds, use_cached = use_cached, path_cached='data/cv_data.pkl')
    print("Folds created")
    print("Load test data")
    test, filenames, _ = dataloader.load_test(filepath='data/test_heat.hdf5', use_cached=use_cached)
    print("Test data loaded")
    # Run everything
    print("Running stacking...")
    st = Stacking(n_folds, stacker_arg, model_arg_list, n_classes)
    predictions = st.fit_predict(data, labels, cval_indx, test)
    print("Saving results...")
    HY.write_submission('stacking_submission', predictions, filenames)
    print("Stacking finished")
