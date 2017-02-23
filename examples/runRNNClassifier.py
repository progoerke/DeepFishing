from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from hyperopt import hp, fmin, tpe

import numpy 
from keras.utils import np_utils

from DeepRNN import DeepRNN

def data():

    syllables = ['aa','ao','ba','bm','ca','ck','da','dl','ea','ej','fa','ff','ha','hk']

    songs = [['aa','bm','ck'],
         ['ao','da','ao','ej'],
         ['ba','ck','ck','dl','ao'],
         ['da','ff','ff'],
         ['ba','ba','fa','fa'],
         ['dl','ha','dl','ha','bm'],
         ['hk','aa','da','hk']]

    nTestSongs = 100
    nTrainRep = 2400
    nvalRep = 240

    max_song_len  = max([len(s) for s in songs])

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    testData = []
    testLabels = []

    SNR = numpy.array([4,2,1,0.5,0.25,0.125])

    scoresPerNum = []
    for l,song in enumerate(songs):
        X_train.extend(int(nTrainRep / len(song)) * [[syllables.index(s) for s in song]])
        X_test.extend(nvalRep * [[syllables.index(s) for s in song]])
        Y_train.extend(int(nTrainRep / len(song)) *[l])
        Y_test.extend(nvalRep *[l])

        testData.extend(nTestSongs * [[syllables.index(s) for s in song]])
        testLabels.extend(nTestSongs *[l])

    X_train = pad_sequences(X_train, maxlen=max_song_len).astype(numpy.float)    
    Y_train = to_categorical(Y_train)      
    X_test = pad_sequences(X_test, maxlen=max_song_len).astype(numpy.float)    
    Y_test = to_categorical(Y_test) 

    testX = pad_sequences(testData, maxlen=max_song_len).astype(numpy.float)  
    testY = to_categorical(testLabels) 

    return X_train, Y_train, X_test, Y_test, SNR, testX, testY


def model(params):


    classifier = DeepRNN(*params)

    acc = classifier.train(X_train, Y_train, X_test, Y_test, SNR)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



if __name__ == '__main__':

    X_train, Y_train, X_val, Y_val, SNR, X_test, Y_test = data()

    space = DeepRNN.space

    best_run = fmin( model, space, algo = tpe.suggest, max_evals = 1 )

    print(best_run)
