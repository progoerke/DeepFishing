from labelBoatIDs import labelBoat
import numpy as np
import sys
import json
import h5py
from skimage import io
import cv2
from keras.preprocessing.image import img_to_array
import glob
import pickle

class_dict = {'ALB':0,'BET':1,'DOL':2,'LAG':3,'NoF':4,'OTHER':5,'SHARK':6,'YFT':7}

#../Pipeline/data/train/ALB/img_00130.jpg
def getClass(filename):
    l = filename.split('/')
    classname = l[4]
    return class_dict[classname]

def getFileName(filename):
    l = filename.split('/')
    return "/".join(l[2:])

def create_train_mat():
    # TRAIN ##

    # For relabelling the training data load the lists of filenames and boatIDs
    y = np.loadtxt('../Pipeline/BoatIDs/img_labels_y.txt')
    f = open('../Pipeline/BoatIDs/img_file_names_unix.json', 'r')
    file_names = json.load(f)
    f.close
    y = y.astype(int)
    y_file_names=[y, file_names]

    num_total_images = 3777

    print('create new hdf5 file')
    file = h5py.File('boat_id_train.hdf5', "w")
    no_chunks = 2
    dt = h5py.special_dtype(vlen=bytes)

    targets = file.create_dataset("targets", (num_total_images,8), chunks=(no_chunks,8), dtype='f', compression="lzf")
    cluster = file.create_dataset("cluster", (num_total_images,1), chunks=(no_chunks,1), dtype='f')
    ids = file.create_dataset("ids", (num_total_images,1), chunks=(no_chunks,1), dtype=dt)

    total = 0

    # Read average images for each cluster
    ncluster = 120
    imgs_averages = [None] * ncluster

    for i in range(len(imgs_averages)):
        imgs_averages[i] = io.imread('../Pipeline/BoatIDs/imgs_averages_' + str(i) + '.jpg')

    for f in file_names:
        img = cv2.imread(f)
        img = np.squeeze(img_to_array(img))
        icluster = labelBoat(img, imgs_averages)
        classlabel = getClass(f)
        targets[total, :] = 0
        targets[total, classlabel] = 1
        cluster[total] = icluster
        ids[total] = getFileName(f)
        total += 1
        if ((total%300) == 0):
           sys.stdout.write(".")
           sys.stdout.flush()
        file.flush()



def create_test_mat():
    # ## TEST ##
    # # For testing data data, only list of filenames
    file_names=glob.glob('data/test_stg1/*.jpg')
    # # Read average images for each cluster

    num_total_images = 1000

    #print('Load test images from file')
    #file_t = h5py.File('test_big.hdf5', "w")
    #images = file['images']
    #ids_test = file['ids']
    #file_t.close()

    print('create new hdf5 file')
    file = h5py.File('boat_id_test.hdf5', "w")
    no_chunks = 2
    dt = h5py.special_dtype(vlen=bytes)

    cluster = file.create_dataset("cluster", (num_total_images,1), chunks=(no_chunks,1), dtype='f')
    ids = file.create_dataset("ids", (num_total_images,1), chunks=(no_chunks,1), dtype=dt)

    ncluster = 120
    imgs_averages = [None] * ncluster

    for i in range(len(imgs_averages)):
        imgs_averages[i] = io.imread('../Pipeline/BoatIDs/imgs_averages_' + str(i) + '.jpg')

    for idx,f in enumerate(file_names):
        img = cv2.imread(f)
        img = np.squeeze(img_to_array(img))
        icluster = labelBoat(img, imgs_averages)
        cluster[idx] = icluster
        ids[idx] = getFileName(f)
        if ((idx%10)==0):
            sys.stdout.write(".")
            sys.stdout.flush()
        if ((idx%100)==0):
            sys.stdout.write("\n")
            sys.stdout.flush()
        file.flush()

def learn_distr():
    print('read training')
    file = h5py.File('boat_id_train.hdf5', "r")
    
    targets = file['targets']
    cluster = file['cluster']
    
    probas = np.zeros((len(np.unique(cluster)),8))
    for idx,cl in enumerate(np.unique(cluster)):
        print('Cluster',cl)
        current_cl_idx = np.where(cluster[:,0] == cl,True,False)
        current_targets = targets[current_cl_idx,:]
        probas[idx,:] = np.sum(current_targets,axis = 0)/np.sum(current_targets)

    print(probas)
    pickle.dump(probas,open('boat_id_probas.pkl','wb'))

def create_prediction():
    print('read test')
    file = h5py.File('boat_id_test.hdf5', "r")
    
    cluster = file['cluster']
    probas = pickle.load(open('boat_id_probas.pkl','rb'))
    print(probas.shape)
    print(cluster.shape)

    predictions = np.zeros((len(cluster),8))
    for i,cl in enumerate(cluster):
        predictions[i,:] = probas[int(cl)-1,:]

    print(predictions)

    pickle.dump(predictions,open('boat_id_pred.pkl','wb'))

def create_prediction_train():
    print('read test')
    file = h5py.File('boat_id_train.hdf5', "r")
    
    cluster = file['cluster']
    probas = pickle.load(open('boat_id_probas.pkl','rb'))
    print(probas.shape)
    print(cluster.shape)

    predictions = np.zeros((len(cluster),8))
    for i,cl in enumerate(cluster):
        predictions[i,:] = probas[int(cl)-1,:]

    print(predictions)

    pickle.dump(predictions,open('boat_id_pred_train.pkl','wb'))
