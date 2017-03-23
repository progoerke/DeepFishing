import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
import cv2
import scipy.misc

print('load from hdf5 file')
file = h5py.File('my_train_mat.hdf5', "r")

images = file["images"]
ids = file["ids"]

for i,img in enumerate(images):
	c_img = img
	# Does not work, why?
	img[0,:,:] = c_img[2,:,:]
	img[2,:,:] = c_img[0,:,:]
	img = (img*255.0).astype(np.uint8)
	name = (ids[i][0].decode("utf-8").split('/'))
	print(name)
	fname = ('').join((name[2],name[3]))
	print(fname)
	scipy.misc.toimage(img, cmin=0.0, cmax=...).save('data/cropped/'+fname)