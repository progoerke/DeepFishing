from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
from  os import listdir
import pickle
import time
from scipy.cluster.vq import whiten

def whiten_all_imgs(path_src,path_dst):
	file_src = h5py.File(path_src, 'r+')
	data = file_src['images']

	file_dst = h5py.File(path_dst, 'w')
	images = file_dst.create_dataset("images", (data.shape), chunks=(64, data.shape[1], data.shape[2], 3), dtype='f', compression="lzf")
	
	# channel 0
	flattened_data = data[:,:,:,0]
	flattened_data = np.resize(flattened_data,(data.shape[0],data.shape[1]*data.shape[2]))
	whitened_data = whiten(flattened_data)
	images[:,:,:,0] = np.resize(whitened_data,(data.shape[0],data.shape[1],data.shape[2]))
	# channel 1
	flattened_data = data[:,:,:,1]
	flattened_data = np.resize(flattened_data,(data.shape[0],data.shape[1]*data.shape[2]))
	whitened_data = whiten(flattened_data)
	images[:,:,:,1] = np.resize(whitened_data,(data.shape[0],data.shape[1],data.shape[2]))
	# channel 2
	flattened_data = data[:,:,:,2]
	flattened_data = np.resize(flattened_data,(data.shape[0],data.shape[1]*data.shape[2]))
	whitened_data = whiten(flattened_data)
	images[:,:,:,2] = np.resize(whitened_data,(data.shape[0],data.shape[1],data.shape[2]))

	file_src.flush()
	file_dst.flush()
	#data[...] = w_data
	#file_src.close()
	#file_dst.close()

#53.42731714248657 all channels
#235.6574468612671 each channel on its own --> not readable
#347.61175203323364 each channel on its own
start = time.time()
filepath = "train_mat.hdf5"
whiten_all_imgs(filepath,'w_train_mat.hdf5')
end = time.time()
print(end - start)
