from labelBoatIDs import labelBoat
import numpy as np
import sys
import json
import h5py
from skimage import io
import cv2
from keras.preprocessing.image import img_to_array


class_dict = {'ALB':0,'BET':1,'DOL':2,'LAG':3,'NoF':4,'OTHER':5,'SHARK':6,'YFT':7}

#../Pipeline/data/train/ALB/img_00130.jpg
def getClass(filename):
    l = filename.split('/')
    classname = l[4]
    return class_dict[classname]

def getFileName(filename):
    l = filename.split('/')
    return "/".join(l[2:])


## TRAIN ##

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




# ## TEST ##
# # For testing data data, only list of filenames
# file_names=glob.glob('/../Pipeline/data/test_stg1/*.jpg')
# # Read average images for each cluster
# ncluster = 120
# imgs_averages = [None] * ncluster

# for i in range(len(imgs_averages)):
#     imgs_averages[i] = io.imread('../Pipeline/BoatIDs/imgs_averages_' + str(i) + '.jpg')

# for f in file_names:
#     img = cv2.imread(f)
#     img = np.squeeze(img_to_array(img))
#     icluster = labelBoat(img, imgs_averages)

#     print('image ', f, ' cluster', icluster)