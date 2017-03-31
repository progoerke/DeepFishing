import pandas as pd
import numpy as np
import glob
from sklearn import cluster
from scipy.misc import imread
import cv2
import skimage.measure as sm
#import progressbar
import multiprocessing
import random
import matplotlib.pyplot as plt
import seaborn as sns


train_files = glob.glob('./input/train/*/*.jpg')
train = np.array([imread(img) for img in train_files])
print('Length of train {}'.format(len(train)))

# Function for computing distance between images
def compare(args):
    img, img2 = args
    img = (img - img.mean()) / img.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.mean(np.abs(img - img2))

# Resize the images to speed it up.
#train = [cv2.resize(img, (224, 224), cv2.INTER_LINEAR) for img in train]

# Create the distance matrix in a multithreaded fashion
pool = multiprocessing.Pool(8)
#bar = progressbar.ProgressBar(max=len(train))
distances = np.zeros((len(train), len(train)))
for i, img in enumerate(train): #enumerate(bar(train)):
    all_imgs = [(img, f) for f in train]
    dists = pool.map(compare, all_imgs)
    distances[i, :] = dists

cls = cluster.DBSCAN(metric='precomputed', min_samples=5, eps=0.6)
y = cls.fit_predict(distances)
print(y)
print('Cluster sizes:')
print(pd.Series(y).value_counts())