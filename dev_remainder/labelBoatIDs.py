
import numpy as np
import glob
import sys
import os
from sklearn import cluster
from scipy.misc import imread
import cv2
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage import io
import json
import statistics


def labelBoat(img, imgs_averages):

    ximg = len(img)
    yimg = len(img[0])

    distances = np.zeros(len(imgs_averages))
    img_bw=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(len(imgs_averages)):
        ximgs = len(imgs_averages[i])
        yimgs = len(sum(imgs_averages[i]))
        if ximg == ximgs and yimg == yimgs:
            img_average_bw = cv2.cvtColor(imgs_averages[i], cv2.COLOR_BGR2GRAY)
            diff = abs(img_bw - img_average_bw)
            dist = sum(sum(diff)) / ximg / yimg
            distances[i] = dist
        else:
            distances[i] = 1000
    icluster = np.argmin(distances)

    return icluster


'''
The code below is to evaluate the results and to show the use of the function above
'''

#file_names=glob.glob('./data/train_all/ALB/*.jpg')
#file_names+=glob.glob('./data/train_all/BET/*.jpg')
#file_names+=glob.glob('./data/train_all/DOL/*.jpg')
#file_names+=glob.glob('./data/train_all/LAG/*.jpg')
#file_names+=glob.glob('./data/train_all/NoF/*.jpg')
#file_names+=glob.glob('./data/train_all/OTHER/*.jpg')
#file_names+=glob.glob('./data/train_all/SHARK/*.jpg')
#file_names+=glob.glob('./data/train_all/YFT/*.jpg')

phase = 'training' # training or testing

if phase == 'training':
    # For relabelling the training data load the lists of filenames and boatIDs
    y = np.loadtxt('../Pipeline/BoatIDs/img_labels_y.txt')
    f = open('../Pipeline/BoatIDs/img_file_names_unix.json', 'r')
    file_names = json.load(f)
    f.close
    y = y.astype(int)
    y_file_names=[y, file_names]
else:
    # For testing data data, only list of filenames
    file_names=glob.glob('/../Pipeline/data/test_stg1/*.jpg')

# Read average images for each cluster
ncluster = 120
imgs_averages = [None] * ncluster

for i in range(len(imgs_averages)):
    imgs_averages[i] = io.imread('../Pipeline/BoatIDs/imgs_averages_' + str(i) + '.jpg')

for f in file_names:
    print(f)
    img = cv2.imread(f)
    img = np.squeeze(img_to_array(img))
    icluster = labelBoat(img, imgs_averages)
    if phase == 'training':
        icluster_orig = y_file_names[0][y_file_names[1].index(f)]
        print('image ',f,' cluster', icluster, ' original cluster', icluster_orig)
    else:
        print('image ', f, ' cluster', icluster)

