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



'''
Load data and split into partitions for cross validation
'''
def data():
    
    cval_splits = 5

    data, labels, _, _ = dataloader.load_train(use_chached=True, use_heatmap=True)
    print('loaded images')
    print('start cross validation')
    cval_indx = CV.VGG_CV(data, labels, folds=cval_splits, use_cached=True)
    print('finished cross validation')
    indx = [np.where(cval_indx==ind) for ind in np.unique(cval_indx)]

    train_indx = np.hstack(indx[:-1])[0].tolist()
    train_indx.sort()
    print('Split train')
    X_train = data[train_indx]
    Y_train = labels[train_indx]

    val_indx = indx[-1][0].tolist()
    val_indx.sort()
    print('Split validation')
    X_val = data[val_indx]
    Y_val = labels[val_indx]

    return X_train, Y_train, X_val, Y_val



'''
Wrapper function to run model training and evaluation
@params - model parameters to optimize
'''
def run_model(params):  

    global best
    global model

    print(params)
    classifier = CCN((X_train.shape[1], X_train.shape[2]),8,12,*params)

    classifier.fit(X_train, Y_train, X_val, Y_val)
    loss = classifier.evaluate(X_val, Y_val)

    if loss < best:
        best = loss
        model = classifier.model
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
            f.write('%s,%s\n' % (os.path.basename(str(image_name)), ','.join(pred)))
        print("Done.")

if __name__ == '__main__':

    global best
    global model
    best = np.inf
    max_evals = 1

    X_train, Y_train, X_val, Y_val = data()

    space = CCN.space
    print('start optimization')
    best_run = fmin(run_model, space, algo = tpe.suggest, max_evals = max_evals)


    print(best_run)
    pickle.dump(best_run, open('models/ccn_loss-{}.pkl'.format(best), 'wb'))
    model.save('models/ccn_loss-{}.h5'.format(best))
    
    #model = load_model('models/ccn_loss-8.677305794018922.h5')
    test, filenames, _ = dataloader.load_test(use_chached=True, use_heatmap=True)
    print(filenames) 
    preds = model.predict(test)
    write_submission(preds, filenames)
