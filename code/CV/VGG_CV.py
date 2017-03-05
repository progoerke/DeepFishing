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
from collections import defaultdict
import json

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


def preprocess_image(path, IMG_HEIGHT, IMG_WIDTH):
    img = image.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


#
# data_path Path to training data.
# n_clusts Number of clusters for K-means
# folds Number of folds for cross-validation
# seed  Seed for the random generator.
def VGG_CV(data_path = './input/train', n_clusts = 5, folds = 5, use_cached=False, path_cached='./cv/cv_data.json'):

    # Path where the CV data will be stored
    cv_store = './cv/cv_data.json'

    if use_cached:
        with open(path_cached, 'r') as data_file:
            data = json.load(data_file)
            cv_data = defaultdict(lambda  : defaultdict(list))
            cont = 0
            for cv in data:
                cv_data[cont][0].append(data[cont]['data']['0'])
                cv_data[cont][1].append(data[cont]['data']['1'])
                cont +=1
            return data

    # Normalized image size for the VGG16 model
    IMG_WIDTH = 640
    IMG_HEIGHT = 360

    # Model
    base_model = VGG16(weights = 'imagenet', include_top = False)#, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    model = Model(input = base_model.input, output = base_model.get_layer('block4_pool').output)

    # how many images to take?
#    SAMP_SIZE = 20

    cv_data = defaultdict(lambda  : defaultdict(list))

    # Subsample data per class
    for fish in os.listdir(data_path):
        # Just checking the data is still there.
        if(os.path.isfile(os.path.join(data_path, fish))):
            continue

        # Get pictures path
        class_pictures = [os.path.join(data_path, fish, fn) for
                          fn in os.listdir(os.path.join(data_path, fish))]
#        class_pictures = class_pictures[:SAMP_SIZE]

        # Get features
        preprocessed_images = np.vstack([preprocess_image(fn, IMG_HEIGHT, IMG_WIDTH) for fn in class_pictures])
        vgg_features = model.predict(preprocessed_images)
        vgg_features = vgg_features.reshape(len(class_pictures), -1)

        # Cluster
        km = KMeans(n_clusters = n_clusts, n_jobs = -1)
        clust_preds = km.fit_predict(StandardScaler().fit_transform(vgg_features))

        # Group all picture paths according to their cluster
        tmp = dict(zip(class_pictures, clust_preds))
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
                    new_train = []
                    new_test = []
                    for ind in train:
                        new_train.append([instances[ind], fish])
                    for ind in test:
                        new_test.append([instances[ind], fish])

                    cv_data[fold][0].extend(new_train)
                    cv_data[fold][1].extend(new_test)
                    fold+=1
            else:   # If we have less instances than clusters
                for instance in instances:
                    fold = randint(0, folds-1)
                    for i in range(0, folds):
                        if i == fold:
                            cv_data[i][1].extend([instance, fish])
                        else:
                            cv_data[i][0].extend([instance, fish])


    # Save data to JSON
    with open('./cv/cv_data.json', 'w') as outfile:
        json.dump([{'cv': k, 'data': v} for k,v in cv_data.items()], outfile, indent=4)

    return cv_data

#VGG_CV()
#VGG_CV(use_cached=True)