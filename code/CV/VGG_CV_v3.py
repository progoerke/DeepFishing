import h5py
import os

from scipy.misc import imread, imsave

from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

import shutil
import numpy as np
from random import randint

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

#path to training data
DATA_PATH = './input/train'

#Number of clusters for K-Means
N_CLUSTS = 5

#Number of clusters used for validation
N_VAL_CLUSTS = 5

#Number of cross-validation folds
folds = 5

SEED = 1
np.random.seed(SEED)

##############################################
#######NORMALIZED IMAGE SIZE
##############################################
IMG_WIDTH = 640
IMG_HEIGHT = 360

#########################
# MODEL

# base_model = VGG16(weights = None, include_top = False, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
base_model = VGG16(weights = 'imagenet', include_top = False)#, input_shape = (IMG_HEIGHT, IMG_WIDTH, 3))
model = Model(input = base_model.input, output = base_model.get_layer('block4_pool').output)

########################
# Preproccess

def preprocess_image(path):
    img = image.load_img(path, target_size = (IMG_HEIGHT, IMG_WIDTH))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis = 0)
    return preprocess_input(arr)

##############################################
#######SUBSAMPLE DATA
##############################################

#how many images to take?
SAMP_SIZE = 20

for fish in os.listdir(DATA_PATH):
    class_pictures = []
    if(os.path.isfile(os.path.join(DATA_PATH, fish))):
        continue

    n_fish = len(os.listdir(os.path.join(DATA_PATH, fish)))  # Number of fish pictures
    fish_per_cv = n_fish/float(folds)

    class_pictures = [os.path.join(DATA_PATH, fish, fn) for
                       fn in os.listdir(os.path.join(DATA_PATH, fish))]
    class_pictures = class_pictures[:SAMP_SIZE]

    # time
    preprocessed_images = np.vstack([preprocess_image(fn) for fn in class_pictures])
    vgg_features = model.predict(preprocessed_images)
    vgg_features = vgg_features.reshape(len(class_pictures), -1)

    # time
    km = KMeans(n_clusters = N_CLUSTS, n_jobs = -1)
    clust_preds = km.fit_predict(StandardScaler().fit_transform(vgg_features))

    tmp = dict(zip(class_pictures, clust_preds))
    inst = {}
    for k, v in tmp.items():
        inst.setdefault(v, []).append(k)

    for i in range(0, N_CLUSTS):
        instances = inst[i]
        if len(instances) >= folds:
            kf = KFold(n_splits=folds)
            fold = -1
            for train, test in kf.split(instances):
                fold += 1
                for index_inst in test:
                    save_path = os.path.join("./input/cv/", str(fold), fish)
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    # print save_path
                    save_path = os.path.join(save_path, os.path.basename(instances[index_inst]))
                    #        imsave(save_path, img)
                    #        tmp1 = save_path
                    #        tmp2 = os.path.join(DATA_PATH,img)
                    shutil.copy2(instances[index_inst], save_path)
        else:
            for instance in instances:
                fold = randint(0,folds-1);
                save_path = os.path.join("./input/cv/", str(fold), fish)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                # print save_path
                save_path = os.path.join(save_path, os.path.basename(instance))
                #        imsave(save_path, img)
                #        tmp1 = save_path
                #        tmp2 = os.path.join(DATA_PATH,img)
                shutil.copy2(instance, save_path)



print("Hello")
input()