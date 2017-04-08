import dataloader
import numpy as np
import hyper

from Classifiers.inceptionV3 import Inception

if __name__ == '__main__':
        
    data, labels, _, _ = dataloader.load_train(filepath='data/train.hdf5',use_cached=True, mode='resize')


    num_val = int(0.1 * len(data)) 
    num_classes = 8
    for c in range(7,num_classes):

        new_labels = np.zeros((len(labels),2))
        ids_class = np.where(labels[:,c] == 1,True,False)
            
        new_labels[ids_class,1] = 1 
        new_labels[~ids_class,0] = 1

        indx = np.arange(len(data))
        np.random.shuffle(indx)
        val_indx = indx[:num_val]
        train_indx = indx[num_val:]        
        
        classifier = Inception((data[0].shape[0],data[0].shape[1]),2,50,0.166,64,'sgd')
        hyper.run_model(m=classifier, data=data, labels=new_labels, train_indx=train_indx, val_indx=val_indx, name='class_{}'.format(c))  

        
