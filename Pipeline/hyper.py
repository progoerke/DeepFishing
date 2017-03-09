import numpy as np
import pickle

import sys
import os

import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from hyperopt import Trials, STATUS_OK, tpe
from hyperopt import hp, fmin, tpe

from code.data_harvesting import heatmap_VisCAM
from code.data_harvesting import dataloader
from code.CV import VGG_CV as CV


#Insert your class here
from Classifiers.CCN import CCN



'''
Load data and split into partitions for cross validation
'''
def data():
    
    cval_splits = 5

    data, labels, _, _ = dataloader.load_train(use_chached=True, use_heatmap=True)

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


if __name__ == '__main__':

    best = np.inf
    max_evals = 100

    X_train, Y_train, X_val, Y_val = data()

    space = CCN.space

    best_run = fmin(run_model, space, algo = tpe.suggest, max_evals = max_evals)

    print(best_run)
    pickle.dumb(best_run, open('optResults.pkl', 'wb'))
