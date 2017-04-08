import hyper as HY
import dataloader
import pickle
import numpy as np
from Classifiers.inceptionV3 import Inception

from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.callbacks import ModelCheckpoint

def create_predictions(modellist):
    test, test_filenames, _ = dataloader.load_test(filepath='data/test.hdf5', directories = 'data/test_stg1',use_cached=True, mode="resize")
    train, targets, train_filenames, _ = dataloader.load_train(filepath='data/train.hdf5', directories = 'data/train',use_cached=True, mode="resize")
    pickle.dump(targets[:],open('predictions/targets.pkl','wb'))
    pickle.dump(test_filenames[:],open('predictions/test_filenames.pkl','wb'))
    pickle.dump(train_filenames[:],open('predictions/train_filenames.pkl','wb'))
    for m in modellist:
        
        model = Inception((train[0].shape[0], train[0].shape[1]),8,100,lr=1e-4,batch_size=64, optimizer='sgd')
        model.model.load_weights('models/{}.h5'.format(m))

        preds = model.predict(test)
        pickle.dump(preds,open('predictions/test_{}.pkl'.format(m),'wb'))
        preds = model.predict(train)
        pickle.dump(preds,open('predictions/train_{}.pkl'.format(m),'wb'))

def train_ensemble(modellist):
    train = np.zeros((len(modellist),3777,8))
    for i,m in enumerate(modellist):
        train[i,:,:] = pickle.load(open('predictions/train_{}.pkl'.format(m),'rb'))

    train = np.transpose(train,(1,0,2))
    dims = train.shape
    train = np.reshape(train,(dims[0],dims[1]*dims[2]))
    targets = pickle.load(open('predictions/targets.pkl','rb'))

    model = Sequential()
    model.add(Dense(1024, input_shape=(len(modellist)*8,)))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())

    # define the checkpoint
    filepath= 'predictions/ensemble_model.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, verbose=1)
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit(train, targets, nb_epoch=100, batch_size=32, callbacks= callbacks_list)


def get_test_predictions(modellist, mode='mean'):
    filenames = pickle.load(open('predictions/test_filenames.pkl','rb'))
    test = np.zeros((len(modellist),1000,8))
    for i,m in enumerate(modellist):
        test[i,:,:] = pickle.load(open('predictions/test_{}.pkl'.format(m),'rb'))

    test = np.transpose(test,(1,0,2))
    print(test.shape)
    if mode=='network':
        
        dims = test.shape
        test = np.reshape(test,(dims[0],dims[1]*dims[2]))

        model = Sequential()
        model.add(Dense(1024, input_shape=(len(modellist)*8,)))
        model.add(Dense(8, activation='softmax'))

        # load the network weights
        filename = "predictions/ensemble_model.hdf5"
        model.load_weights(filename)
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        print(model.summary())

        predictions = model.predict(test)
        print(predictions.shape)
    elif mode=="mean":
        predictions = np.mean(test,axis=1)
        print(predictions.shape)
    elif mode=='max':
        predictions = np.max(test,axis=1)
        print(predictions.shape)

    HY.write_submission('ensemble_submission', predictions, filenames)


if __name__ == '__main__':

    modellist = ['class_0-1.1920930376163597e-07','class_1-1.1920930376163597e-07','class_2-1.1920930376163597e-07','class_3-1.1920930376163597e-07','class_4-1.1920930376163597e-07','class_5-1.1920930376163597e-07','class_6-1.1920930376163597e-07','class_7-1.1920930376163597e-07']
    #create_predictions(modellist)
    #train_ensemble(modellist)
    get_test_predictions(modellist, mode='max')

    
