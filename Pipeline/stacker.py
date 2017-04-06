import dataloader
import pickle
import numpy as np
from Classifiers.inceptionV3 import Inception
import hyper as HY

from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

import VGG_CV as CV
from keras.optimizers import SGD
from hyper import train_generator
from keras.preprocessing.image import ImageDataGenerator

###################################################################################
###################################################################################

class Stacking(object):
    def __init__(self, n_folds, stacker_args, base_models_args):
        self.n_folds = n_folds
        # Check that the number of folds correspond to the number of base_models
        if n_folds != len(base_models_args):
            raise Warning("The number of folds does no correspond to the number of base models.\n"
                          "Using just the first {} base models".format(n_folds))
        self.stacker_args = stacker_args
        self.base_models_args = base_models_args[:n_folds]

    def fit_predict(self, data, labels, cval_indx ,test):
        n_classes = 8

        s_train = np.zeros((data.shape[0], len(self.base_models_args)*n_classes))
        s_test = np.zeros((test.shape[0], len(self.base_models_args)*n_classes))
        print(len(self.base_models_args))
        for i in range(len(self.base_models_args)):
            s_test_i = np.zeros((self.n_folds, test.shape[0], 8))
            for j in range(self.n_folds):
                # Divide training and validation
                indx = [np.where(cval_indx == ind) for ind in np.unique(cval_indx)]
                selector = [x for k,x in enumerate(indx) if k != j][0]
                train_indx = selector[0].tolist()
                train_indx.sort()
                selector = [x for k, x in enumerate(indx) if k == j][0]
                val_indx = selector[0].tolist()
                val_indx.sort()
                # Load the model
                base_model = Inception((data[0].shape[0], data[0].shape[1]), 8, 1, *self.base_models_args[i])
                # Train
                model = self.train_model(classifier=base_model, data=data, labels=labels, train_indx=train_indx, val_indx=val_indx)
                #Get predictions
                val_set = np.array([data[k] for k in val_indx])
                y_pred = base_model.predict(val_set)
                s_train[val_indx, i*n_classes:i*n_classes+n_classes] = y_pred
                s_test_i[j,:,:] = model.predict(test)


            tmp = np.mean( np.array(s_test_i), axis=0 )
            s_test[:, i*n_classes:i*n_classes+8] = tmp

        # Create stacker
        stacker = Sequential()
        stacker.add(Dense(1024, input_shape=(len(self.base_models_args) * 8,)))
        stacker.add(Dense(8, activation='softmax'))
        stacker.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        print(stacker.summary())

        # define the checkpoint
        filepath = 'models/stacker_model_loss-{loss:.4f}.hdf5'
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,save_best_only=True)
        callbacks_list = [checkpoint]

        # Fit the model

        stacker.fit(s_train, labels, nb_epoch=10, batch_size=32, callbacks=callbacks_list, shuffle='batch')
        y_pred = stacker.predict(s_test)
        return y_pred

    '''
    Wrapper function to run model training and evaluation
    @params - model parameters to optimize
    '''

    def train_model(self, classifier=None, data=None, labels=None, train_indx=None, val_indx=None):

        # classifier.create_class_weight(dict(enumerate(np.sum(labels,0))))

        tg = train_generator(data, labels, train_indx, classifier.batch_size)
        vg = train_generator(data, labels, val_indx, classifier.batch_size)

        classifier.fit(tg, vg, (len(train_indx), len(val_indx)))

        for layer in classifier.model.layers[:172]:
            layer.trainable = False
        for layer in classifier.model.layers[172:]:
            layer.trainable = True

        classifier.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
                                 metrics=['accuracy'])

        classifier.fit(tg, vg, (len(train_indx), len(val_indx)))

        return classifier


###################################################################################
###################################################################################

if __name__ == '__main__':
    model_arg_files = ['models/resnet_loss-1.4480618794978652.pkl',
                       'models/resnet_loss-1.9494444401883289.pkl']
    stacker_arg_file = []
    n_folds = 2
    use_cached = True
    use_heatmap = False
    # Read parameters from file
    print("Reading parameters for the models...")
    model_arg_list = []
    stacker_arg = []
    for file in model_arg_files:
        model_arg_list.append(pickle.load(open(file, "rb")))
    # stacker_arg = pickle.load(open(stacker_arg_file, "rb"))
    print("Parameters read")
    # Prepare data
    print("Loading training data...")
    data, labels, _, _ = dataloader.load_train(filepath='data/train.hdf5',use_cached=use_cached)
    print("Training data loaded")
    print("Creating {}-folds...".format(n_folds))
    cval_indx = CV.VGG_CV(data, labels, folds=n_folds, use_cached = True)
    print("Folds created")
    print("Load test data")
    test, filenames, _ = dataloader.load_test(filepath='data/test_stg1.hdf5',use_cached=False) # use_cached=use_cached
    print("Test data loaded")
    # Run everything
    print("Running stacking...")
    st = Stacking(n_folds, stacker_arg, model_arg_list)
    predictions = st.fit_predict(data, labels, cval_indx, test)
    print("Saving results...")
    HY.write_submission('stacking_submission', predictions, filenames)
    print("Stacking finished")
