import os

from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

import numpy as np
from random import randint
import pickle 
import collections

# Stratified Cross-Validation approach.
#
# It first uses the 4th layer of a VGG network to create clusters. Then, according to those
# clusters, instances are divided into k-folds so in each fold there are approximately the same number of instances of
# each cluster.
#
# For the formation of clusters only pictures of one fish class are used, so the proccess is repeated as many times as
# classes we have.
#
# Original idea from https://www.kaggle.com/dollardollar/the-nature-conservancy-fisheries-monitoring/validation-split-via-vgg-based-clustering


def preprocess_image(img):
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


#
# data_path Path to training data.
# n_clusts Number of clusters for K-means
# folds Number of folds for cross-validation
# seed  Seed for the random generator.
#
# Outputs list of tuples [index - fold]
def VGG_CV(data, data_labels, img_height = 400, img_width=400, n_clusts = 5, folds = 5, use_cached=False, path_cached='./cv/cv_data.json'):

    # Path where the CV data will be stored
    cv_store = './cv_data.pkl'

#    if use_cached:
#        with open(path_cached, 'r') as data_file:
#            data = json.load(data_file)
#            cv_data = defaultdict(lambda  : defaultdict(list))
#             cont = 0
#            for cv in data:
#                cv_data[cont][0].append(data[cont]['data']['0'])
#                cv_data[cont][1].append(data[cont]['data']['1'])
#                cont +=1
#            return data

    # Model
    base_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (img_height, img_width, 3))
    model = Model(input = base_model.input, output = base_model.get_layer('block4_pool').output)

    cv_data = np.zeros(len(data))

    fish_per_label = np.sum(data_labels,axis=0)
    print(fish_per_label)
    # Subsample data per class
    for fish in range(data_labels.shape[1]):

        num_pictures = fish_per_label[fish]
        
        if not num_pictures:
            continue

        indexes = np.empty(num_pictures)
        cont = 0
        for ind in range(0,len(data)):
            if data_labels[ind][fish] == 1:
                indexes[cont] = ind
                cont+=1

        # Get pictures
        class_pictures = np.empty([num_pictures,img_height, img_width, 3])
        cont1 = 0
        cont2 = 0
        for label in data_labels:
            if label[fish] == 1:
                class_pictures[cont1] = data[cont2]
                cont1+=1
            cont2+=1

        # Get features
        preprocessed_images = np.vstack([preprocess_image(fn) for fn in class_pictures])
        vgg_features = model.predict(preprocessed_images)
        vgg_features = vgg_features.reshape(len(data), -1)

        # Cluster
        km = KMeans(n_clusters = n_clusts, n_jobs = -1)
        clust_preds = km.fit_predict(StandardScaler().fit_transform(vgg_features))

        # Group all picture according to their cluster
        tmp = dict(zip(indexes, clust_preds))
        inst = {}
        for k, v in tmp.items():
            inst.setdefault(v, []).append(k)

        # For each cluster
        for i in range(0, n_clusts):
            instances = inst[i]
            if len(instances) >= folds: # If we have more instances than clusters
                # Divide in train/validation sets
                kf = KFold(n_splits=folds)
                fold = 0
                for train, test in kf.split(instances):
                    for ind in test:
                        cv_data[int(instances[ind])] = fold
                    fold+=1
            else:   # If we have less instances than clusters
                for instance in instances:
                    fold = randint(0, folds-1)
                    for i in range(0, folds):
                        if i == fold:
                            cv_data[int(instance)] = fold

    # Save data to JSON
    with open(cv_store, 'wb') as outfile:
        pickle.dump(cv_data, outfile)

    return cv_data

#VGG_CV()
#VGG_CV(use_cached=True)