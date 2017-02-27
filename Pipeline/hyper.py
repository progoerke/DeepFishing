import numpy as np
import pickle

from hyperopt import Trials, STATUS_OK, tpe
from hyperopt import hp, fmin, tpe


import dataLoader
import crossValidation

#Insert your class here
from yourClassifier import yourClassifier

'''
Load data and split into partitions for cross validation
'''
def data():
	
	cval_splits = 10

	data, labels = dataLoader.load()
	cval_indx = crossValidation.split(data, cval_splits)

	train_indx = cval_indx[:-2]
	X_train = data[train_indx]
	Y_train = label[train_indx]

	val_indx = cval_indx[-1]
	X_val = data[val_indx]
	Y_val = data[val_indx]

	return X_train, Y_train, X_val, Y_val



'''
Wrapper function to run model training and evaluation
@params - model parameters to optimize
'''
def run_model(params):	

	classifier = yourClassifier(*params)

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

	space = yourClassifier.space

	best_run = fmin(model, space, algo = tpe.suggest, max_evals = max_evals)

    print(best_run)
    pickle.dumb(best_run, open('optResults.pkl', 'wb'))
