import numpy as np
from keras.utils import np_utils
import utils
from CCN import CCN

def data():
    nb_classes = 8
    train_data, train_target, train_id = utils.load_train()
    test_data, test_id = utils.load_test()

    X_train = np.array(train_data, dtype=np.uint8)
    y_train = np.array(train_target, dtype=np.uint8)
    X_train = X_train.transpose((0, 3, 1, 2))
    Y_train = np_utils.to_categorical(y_train, nb_classes)

    X_test = np.array(test_data, dtype=np.uint8)
    X_test = X_test.transpose((0, 3, 1, 2))

    # Normalizing the data
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train, Y_train, X_test

def model(X_train, Y_train, X_test):
    c = CCN()

    c.build()

    c.fit(X_train, Y_train)

    c.predict(X_test)

if __name__ == '__main__':
    X_train, Y_train, X_test = data()

    model( X_train, Y_train, X_test)




