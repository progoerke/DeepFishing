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


#Insert your class here
from Classifiers.CCN import CCN



'''
Load data and split into partitions for cross validation
'''
def data():
    
    cval_splits = 5

    data, labels, _, _ = dataloader.load_train(use_chached=False, use_heatmap=True)

    cval_indx = CV.VGG_CV(data, labels, folds=cval_splits)
    indx = [np.where(cval_indx==ind) for ind in np.unique(cval_indx)]

    train_indx = np.hstack(indx[:-1])
    X_train = data[train_indx]
    Y_train = labels[train_indx]

    val_indx = indx[-1]
    X_val = data[val_indx]
    Y_val = labels[val_indx]

    return X_train, Y_train, X_val, Y_val



'''
Wrapper function to run model training and evaluation
@params - model parameters to optimize
'''
def run_model(params):  

    classifier = CCN(size = (32, 32), nb_classes = 8, nb_epoch = 12, *params)

    classifier.fit(X_train, Y_train, X_val, Y_val)
    loss = classifier.evaluate(X_val, Y_val)

    if loss < best:
        best = loss 
        print('new best: ', best, params)

    return {'loss': loss, 'status': STATUS_OK, 'model': model}


def write_submission(predictions, filenames):
    preds = np.clip(predictions, 0.01, 1-0.01)
    sub_fn = 'data/first' 
    
    with open(sub_fn + '.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.6f' % p for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
        print("Done.")

if __name__ == '__main__':

    best = np.inf
    max_evals = 1

    X_train, Y_train, X_val, Y_val = data()

    space = CCN.space

    best_run = fmin(run_model, space, algo = tpe.suggest, max_evals = max_evals)


    print(best_run)
    pickle.dumb(best_run, open('optResults.pkl', 'wb'))

    test, filenames, _ = dataloader.load_test(use_chached=False, use_heatmap=True)
    
    classifier = CCN(size = (32,32), nb_classes = 8, nb_epoch = 12, **best_run)
    preds = classifier.predict(test)

    write_submission(preds, filenames)
