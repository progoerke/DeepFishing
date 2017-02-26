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
file = h5py.File("data_mat.hdf5",'r')
#a = file["X"][()]
a = file["images"]
all_imgs = np.array(a)
file.close()
print(all_imgs.shape)
#all_imgs_whitened = whiten_imgs(all_imgs[0:10,:,:,:])
end = time.time()
print(end - start)
zer = np.zeros((200,200,3))
zer[:,:,0] = all_imgs[0,0,:,:]
zer[:,:,1] = all_imgs[0,1,:,:]
zer[:,:,2] = all_imgs[0,2,:,:]

zer = zer[:, :, ::-1]

zer2 = np.zeros((200,200,3))
zer2[:,:,0] = all_imgs[1,0,:,:]
zer2[:,:,1] = all_imgs[1,1,:,:]
zer2[:,:,2] = all_imgs[1,2,:,:]

zer2 = zer2[:, :, ::-1]

fig, ax = plt.subplots(2,1)
ax[0].imshow(zer)
ax[1].imshow(zer2)
# zer = np.zeros((300,300,3))
# zer[:,:,0] = all_imgs[0,:,:,2]
# ax[0,0].imshow(zer)
# zer = np.zeros((300,300,3))
# zer[:,:,1] = all_imgs[0,:,:,1]
# ax[0,1].imshow(zer)
# zer = np.zeros((300,300,3))
# zer[:,:,2] = all_imgs[0,:,:,0]
# ax[0,2].imshow(zer)
# zer[:,:,2] = all_imgs[0,:,:,0]
# zer[:,:,1] = all_imgs[0,:,:,1]
# zer[:,:,0] = all_imgs[0,:,:,2]
# ax[0,3].imshow(zer)
# zer = np.zeros((300,300,3))
# zer[:,:,0] = all_imgs[0,:,:,0]
# ax[0,0].imshow(zer.astype(float))
# zer = np.zeros((300,300,3))
# zer[:,:,1] = all_imgs[0,:,:,1]
# ax[0,1].imshow(zer.astype(float))
# zer = np.zeros((300,300,3))
# zer[:,:,2] = all_imgs[0,:,:,2]
# ax[0,2].imshow(zer.astype(float))
# zer[:,:,2] = all_imgs[0,:,:,2]
# zer[:,:,1] = all_imgs[0,:,:,1]
# zer[:,:,0] = all_imgs[0,:,:,0]
# print(np.min(zer))
# print(np.max(zer))
# ax[0,3].imshow(zer.astype(float))


# zer = np.zeros((300,300,3))
# zer[:,:,0] = all_imgs_whitened[0,:,:,0]
# ax[1,0].imshow(zer)
# zer = np.zeros((300,300,3))
# zer[:,:,1] = all_imgs_whitened[0,:,:,1]
# ax[1,1].imshow(zer)
# zer = np.zeros((300,300,3))
# zer[:,:,2] = all_imgs_whitened[0,:,:,2]
# ax[1,2].imshow(zer)
# zer = np.zeros((300,300,3))
# ax[1,3].imshow(all_imgs_whitened[0])

plt.show()

# 200,200 cryptic result, 41,6 sec
# 300,300 with 10 imgs, 23,6 sec
# 300,300 175,1 sec
# 500,500 run out of application memory