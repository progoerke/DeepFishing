import numpy as np
import pickle

import sys
import os

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from hyperopt import Trials, STATUS_OK
from hyperopt import hp, fmin, tpe

import heatmap_VisCAM
import dataloader
import VGG_CV as CV

from keras.models import load_model

#Insert your class here
from Classifiers.CCN import CCN
from Classifiers.inceptionV3 import Inception


'''
Load data and split into partitions for cross validation
@param load: if true entire data is laoded into memory
'''
def data(load=False, use_cached=True, use_heatmap=True):
    
    cval_splits = 5

    data, labels, _, _ = dataloader.load_train(filepath='/work/kstandvoss/train_mat.hdf5',use_chached=use_cached, use_heatmap=use_heatmap)
    print('loaded images')
    print('start cross validation')
    cval_indx = CV.VGG_CV(data, labels, folds=cval_splits, use_cached=use_cached)
    print('finished cross validation')
    indx = [np.where(cval_indx==ind) for ind in np.unique(cval_indx)]
    
    train_indx = np.hstack(indx[:-1])[0].tolist()
    train_indx.sort()

    val_indx = indx[-1][0].tolist()
    val_indx.sort()

    if load:
        data = data[:]
        labels = labels[:]

    return data, labels, train_indx, val_indx


def train_generator(data, labels, train_indx, batch_size):

    np.random.shuffle(train_indx)
    start = 0
    while True:
        indx = train_indx[start:start+batch_size]
        start += batch_size
        start %= len(train_indx)

        yield (data[indx], labels[indx])                

def val_generator(data, labels, val_indx, batch_size):
    
    np.random.shuffle(val_indx)
    start = 0
    while True:
        indx = train_indx[start:start+batch_size]
        start += batch_size 
        start %= len(val_indx)

        yield (data[indx], labels[indx])

'''
Wrapper function to run model training and evaluation
@params - model parameters to optimize
'''
def run_model(params=None, m=None):  

    global best
    global model

    print(params)
    #classifier = CCN((data[0].shape[0], data[0].shape[1]),8,15,*params)
    if params:
        classifier = Inception((data[0].shape[0], data[0].shape[1]),8,50,*params)
    else:
        classifier = m

    classifier.create_class_weight(dict(enumerate(np.sum(labels,0))))

    tg = train_generator(data, labels, train_indx, classifier.batch_size)
    vg = train_generator(data, labels, val_indx, classifier.batch_size)

    classifier.fit(tg, vg, (len(train_indx), len(val_indx)))
    loss = classifier.evaluate(vg, len(val_indx))

    if loss < best:
        best = loss
        model = classifier.model
        pickle.dump(params, open('models/inception_loss-{}.pkl'.format(best), 'wb'))
        model.save_weights('models/inception_loss-{}.h5'.format(best))
        print('new best: ', best, params)

    return {'loss': loss, 'status': STATUS_OK}


def write_submission(predictions, filenames):
    preds = np.clip(predictions, 0.01, 1-0.01)
    sub_fn = 'data/first' 
    
    with open(sub_fn + '.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.6f' % p for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(str(image_name[0],'utf-8')), ','.join(pred)))
        print("Done.")

def optimize(max_evals):
    #space = CCN.space
    space = Inception.space
    print('start optimization')
    best_run = fmin(run_model, space, algo = tpe.suggest, max_evals = max_evals)

    print(best_run)
    pickle.dump(best_run, open('models/inception_loss-{}.pkl'.format(best), 'wb'))
    model.save_weights('models/inception_loss-{}.h5'.format(best))
    

if __name__ == '__main__':


    global best
    global model
    best = np.inf

    

    if sys.argv[1] == '-o':
        data, labels, train_indx, val_indx = data(True, False, False)
        max_evals = int(sys.argv[2])
        optimize(max_evals)
    elif sys.argv[1] == '-r':
        data, labels, train_indx, val_indx = data(False)
        run_model(params)        
    else:
        name = sys.argv[1]
        data, labels, train_indx, val_indx = data(True, True, False)
        params = pickle.load(open('models/{}.pkl'.format(name),'rb'))
        model = Inception((data[0].shape[0], data[0].shape[1]),8,100,lr=params['lr'],batch_size=64, optimizer='sgd')
        model.model.load_weights('models/{}.h5'.format(name))
        model.model.compile(loss='categorical_crossentropy',
                      optimizer=model.optimizer,
                      metrics=["accuracy"])
        run_model(m=model)
    
    test, filenames, _ = dataloader.load_test(filepath='/work/kstandvoss/test_mat.hdf5', use_chached=True, use_heatmap=False)
    print(filenames) 
    preds = model.predict(test)
    write_submission(preds, filenames)
