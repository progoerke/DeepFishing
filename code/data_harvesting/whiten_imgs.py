from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
from  os import listdir
import pickle
import time
from scipy.cluster.vq import whiten

def whiten_imgs(all_imgs):
	all_imgs_flattened = np.resize(all_imgs,(all_imgs.shape[0],all_imgs.shape[1]*all_imgs.shape[2]*all_imgs.shape[3]))
	print(all_imgs_flattened.shape)
	all_imgs_whitened = whiten(all_imgs_flattened)
	all_imgs_whitened = np.resize(all_imgs_whitened,all_imgs.shape)
	return all_imgs_whitened


start = time.time()
file = h5py.File("X_mat.h5py",'r')
a = file["X"]
all_imgs = np.array(a)
file.close()
print(all_imgs.shape)
all_imgs_whitened = whiten_imgs(all_imgs)
end = time.time()
print(end - start)

fig, ax = plt.subplots(2,1)
ax[0].imshow(all_imgs[0])
ax[1].imshow(all_imgs_whitened[0])

plt.show()