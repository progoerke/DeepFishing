
# coding: utf-8

# In[4]:

from keras import layers
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from keras.engine import Layer
from os import listdir
from os.path import isfile, join, dirname
from scipy.io import loadmat
import gc
from keras.models import Model
from keras.layers import *

from vis.visualization import visualize_cam

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import model_from_json


meta_clsloc_file = join(dirname(__file__), "/work/kstandvoss/data", "meta_clsloc.mat")
synsets = loadmat(meta_clsloc_file)["synsets"][0]
synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],
                                 key=lambda v:v[1])

corr = {}
for j in range(1000):
    corr[synsets_imagenet_sorted[j][0]] = j

corr_inv = {}
for j in range(1,1001):
    corr_inv[corr[j]] = j

def depthfirstsearch(id_, out=None):
    if out is None:
        out = []
    if isinstance(id_, int):
        pass
    else:
        id_ = next(int(s[0]) for s in synsets if s[1][0] == id_)

    out.append(id_)
    children = synsets[id_ - 1][5][0]
    for c in children:
        depthfirstsearch(int(c), out)
    return out

# This is to find all the outputs that correspond to the class we want.
def synset_to_dfs_ids(synset):
    ids = [x for x in depthfirstsearch(synset) if x <= 1000]
    ids = [corr[x] for x in ids]
    return ids

def layer_type(layer):
    return str(layer)[10:].split(" ")[0].split(".")[-1]

def detect_configuration(model):
    # must return the configuration and the number of the first pooling layer

    # Names (types) of layers from end to beggining
    inverted_list_layers = [layer_type(layer) for layer in model.layers[::-1]]

    layer1 = None
    layer2 = None

    i = len(model.layers)

    for layer in inverted_list_layers:
        i -= 1
        if layer2 is None:
            if layer == "GlobalAveragePooling2D" or layer == "GlobalMaxPooling2D":
                layer2 = layer

            elif layer == "Flatten":
                return "local pooling - flatten", i - 1

        else:
            layer1 = layer
            break

    if layer1 == "MaxPooling2D" and layer2 == "GlobalMaxPooling2D":
        return "local pooling - global pooling (same type)", i
    elif layer1 == "AveragePooling2D" and layer2 == "GlobalAveragePooling2D":
        return "local pooling - global pooling (same type)", i

    elif layer1 == "MaxPooling2D" and layer2 == "GlobalAveragePooling2D":
        return "local pooling - global pooling (different type)", i + 1
    elif layer1 == "AveragePooling2D" and layer2 == "GlobalMaxPooling2D":
        return "local pooling - global pooling (different type)", i + 1

    else:
        return "global pooling", i

