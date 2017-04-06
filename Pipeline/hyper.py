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
from keras.preprocessing.image import ImageDataGenerator

#Insert your class here
from Classifiers.CCN import CCN
from Classifiers.inceptionV3 import Inception
from keras.optimizers import SGD

'''
Load data and split into partitions for cross validation
@param load: if true entire data is loaded into memory
'''
def data(load=False, use_cached=True, use_heatmap=True):
    
    cval_splits = 5
    data, labels, _, _ = dataloader.load_train(filepath='/work/kstandvoss/train_mat_heat.hdf5',directories='data/train',use_cached=use_cached, mode="resize")
    print('loaded images')
    print('start cross validation')
    cval_indx = CV.VGG_CV(data, labels, folds=cval_splits, use_cached=True)
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
    prob_8 = 1/(np.sum(labels,axis=0)+1)
    prob_all = np.zeros(len(train_indx))
    for ind,i in enumerate(train_indx):
        prob_all[ind] = prob_8[labels[i]==1]
    
    prob_all = prob_all/np.sum(prob_all)

    datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    num = 500 
    while True:
        sampled_indx = np.random.choice(train_indx,size=num, p=prob_all)
        d = []
        l = []

        for i in sampled_indx:
            d.append(data[i])
            l.append(labels[i])
        d = np.array(d)
        l = np.array(l)

        datagen.fit(d)

        cnt = 0
        for X_batch, Y_batch in datagen.flow(d,l, batch_size=batch_size):
            weight = np.sum(Y_batch,axis=0) + 1
            weight = np.clip(np.log(np.sum(weight)/weight),1,5)
            weight = np.tile(weight, (len(Y_batch),1))[Y_batch==1] 
            yield (X_batch, Y_batch, weight)
            cnt+=batch_size   
            if cnt == num:
                break
                 

def val_generator(data, labels, val_indx, batch_size):
    
    np.random.shuffle(val_indx)

    start = 0
    while True:
        indx = val_indx[start:start+batch_size]
        start += batch_size 
        start %= len(val_indx)

        yield (data[indx.sort()], labels[indx.sort()])

'''
Wrapper function to run model training and evaluation
@params - model parameters to optimize
'''
def run_model(params=None, m=None, data=None, labels=None, train_indx=None, val_indx=None):  

    global best
    global model

    print(params)
    #classifier = CCN((data[0].shape[0], data[0].shape[1]),8,15,*params)
    if params:
        classifier = Inception((data[0].shape[0], data[0].shape[1]),8,50,*params)
    else:
        classifier = m

    #classifier.create_class_weight(dict(enumerate(np.sum(labels,0))))

    tg = train_generator(data, labels, train_indx, classifier.batch_size)
    vg = train_generator(data, labels, val_indx, classifier.batch_size)

    classifier.fit(tg, vg, (len(train_indx), len(val_indx)))

    for layer in classifier.model.layers[:172]:
            layer.trainable = False
    for layer in classifier.model.layers[172:]:
            layer.trainable = True

    classifier.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])        
    
    classifier.fit(tg, vg, (len(train_indx), len(val_indx)))

    loss = classifier.evaluate(vg, len(val_indx))

    if loss < best:
        best = loss
        model = classifier.model
        pickle.dump(params, open('/work/kstandvoss/models/inception_fine_loss-{}.pkl'.format(best), 'wb'))
        model.save_weights('/work/kstandvoss/models/inception_fine_loss-{}.h5'.format(best))
        print('new best: ', best, params)

        # serialize model to JSON
        model_json = model.to_json()
        with open('/work/kstandvoss/model_json/inception_fine_loss-{}.json'.format(best), "w") as json_file:
            json_file.write(model_json)
        print("Saved model to disk")

    return {'loss': loss, 'status': STATUS_OK}


def write_submission(csv_name, predictions, filenames):
    preds = np.clip(predictions, 0.01, 1-0.01)
    sub_fn = '/work/kstandvoss/data/' + csv_name
    
    with open(sub_fn + '.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.6f' % p for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(str(image_name[0],'utf-8')), ','.join(pred)))
        print("Done.")

def optimize(max_evals, data=None, labels=None, train_indx=None, val_indx=None):
    #space = CCN.space
    space = Inception.space
    print('start optimization')

    run = lambda params: run_model(params, data=data, labels=labels, train_indx=train_indx, val_indx=val_indx)
    best_run = fmin(run, space, algo = tpe.suggest, max_evals = max_evals)

    print(best_run)

    pickle.dump(best_run, open('/work/kstandvoss/models/inception_fine_loss-{}.pkl'.format(best), 'wb'))
    model.save_weights('/work/kstandvoss/models/inception_fine_loss-{}.h5'.format(best))

    # serialize model to JSON
    model_json = model.to_json()
    with open('model_json/inception_fine_loss-{}.json'.format(best), "w") as json_file:
        json_file.write(model_json)
    print("Saved model to disk")
    

if __name__ == '__main__':

    np.random.seed(42)

    global best
    global model
    best = np.inf

    if sys.argv[1] == '-o':
        data, labels, train_indx, val_indx = data(False, True, False)
        max_evals = int(sys.argv[2])
        optimize(max_evals,data=data, labels=labels,train_indx=train_indx,val_indx=val_indx)
    elif sys.argv[1] == '-r':
        data, labels, train_indx, val_indx = data(False, False, False)
        params = pickle.load(open('/work/kstandvoss/models/{}.pkl'.format('inception_loss-0.3928944135109009'),'rb'))
        run_model((params['lr'],64,'sgd'),data=data, labels=labels,train_indx=train_indx,val_indx=val_indx)        
    else:
        name = sys.argv[1]
        data, labels, train_indx, val_indx = data(True, True, False)
        params = pickle.load(open('/work/kstandvoss/models/{}.pkl'.format(name),'rb'))
        model = Inception((data[0].shape[0], data[0].shape[1]),8,100,lr=1e-4,batch_size=64, optimizer='sgd')
        model.model.load_weights('/work/kstandvoss/models/{}.h5'.format(name))

        for layer in model.model.layers[:236]:
            layer.trainable = False
        for layer in model.model.layers[236:]:
            layer.trainable = True
        model.model.compile(loss='categorical_crossentropy',
                      optimizer=model.optimizer,
                      metrics=["accuracy"])
        run_model(m=model,data=data, labels=labels,train_indx=train_indx,val_indx=val_indx)
    
    test, filenames, _ = dataloader.load_test(filepath='/work/kstandvoss/test_mat.hdf5',directories='data/test_stg1', use_cached=True, mode="resize")
    print(filenames) 
    preds = model.predict(test)
    write_submission('first',preds, filenames)
