from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
from  os import listdir
import pickle
import time
from scipy.cluster.vq import whiten

def whiten_all_imgs(path):
	file = h5py.File(path, 'r+')
	data = file['images']
	flattened_data = np.resize(data,(data.shape[0],data.shape[1]*data.shape[2]*data.shape[3]))
	w_data = whiten(flattened_data)
	w_data = np.resize(w_data,data.shape)
	data[...] = w_data
	file.close()

#53.42731714248657
start = time.time()
filepath = "train_mat.hdf5"
whiten_all_imgs(filepath)
end = time.time()
print(end - start)
