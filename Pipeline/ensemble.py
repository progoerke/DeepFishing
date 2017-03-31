import hyper as HY
import dataloader
import pickle
import numpy as np


def create_predictions(modellist):
    test, test_filenames, _ = dataloader.load_test(filepath='/work/kstandvoss/test_mat.hdf5', use_chached=True, use_heatmap=False)
    train, targets, train_filenames, _ = dataloader.load_train(filepath='/work/kstandvoss/train_mat.hdf5', use_chached=True, use_heatmap=False)
    pickle.dump(targets,open('/work/kstandvoss/predictions/targets.pkl','wb'))
    pickle.dump(test_filenames,open('/work/kstandvoss/predictions/test_filenames.pkl','wb'))
    pickle.dump(train_filenames,open('/work/kstandvoss/predictions/train_filenames.pkl','wb'))
    for m in modellist:
        params = pickle.load(open('models/{}.pkl'.format(m),'rb'))
        preds = model.predict(test)
        pickle.dump(preds,open('/work/kstandvoss/predictions/test_{}.pkl'.format(m),'wb'))
        preds = model.predict(train)
        pickle.dump(preds,open('/work/kstandvoss/predictions/train_{}.pkl'.format(m),'wb'))

def train_ensemble(modellist):
    train = np.zeros((len(modellist),3777,8))
    for i,m in enumerate(modellist):
        train[i,:,:] = pickle.load(open('/work/kstandvoss/predictions/train_{}.pkl'.format(m),'rb'))
        targets[i,:,:] = targ

    train = np.transpose(train,(1,0,2))
    targets = pickle.load(open('/work/kstandvoss/predictions/targets.pkl','rb'))

    model = Sequential()
    model.add(Dense(32, input_shape=(len(modellist),8)))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    print(model.summary())

    # define the checkpoint
    filepath= '/work/kstandvoss/predictions/model_loss-{loss:.4f}.hdf5'
    mkdir_p(os.path.dirname(filepath))
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1)
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit(train, targets, nb_epoch=10, batch_size=10, callbacks_list= callbacks_list)


def get_test_predictions(modellist, mode='mean'):
    filenames = pickle.load(open('/work/kstandvoss/predictions/test_filenames.pkl','rb'))
    test = np.zeros((len(modellist),1000,8))
    for i,m in enumerate(modellist):
        test[i,:,:] = pickle.load(open('/work/kstandvoss/predictions/test_{}.pkl'.format(m),'rb'))
    test = np.transpose(train,(1,0,2))

    if mode=='network':
        
        model = Sequential()
        model.add(Dense(32, input_shape=(len(modellist),8)))
        model.add(Dense(8, activation='softmax'))

        # load the network weights
        filename = "/work/kstandvoss/predictions/model_loss-xxxxx.hdf5"
        model.load_weights(filename)
        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        print(model.summary())

        predictions = model.predict(test)
        print(predictions.shape)
    elif mode=="mean":
        predictions = np.mean(test,axis=0)
        print(predictions.shape)
    elif mode=='max':
        predictions = np.max(test,axis=0)
        print(predictions.shape)

    HY.write_submission('ensemble_submission', predictions, filenames)

    
