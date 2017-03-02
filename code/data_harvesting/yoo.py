import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from imgloader import load_single_img
from heatmap_VisCAM import heatmap

no = 1

#heatmap(current_img)


#def rotate_fish():
# Load all fish from hdf5

images, targets, ids, crop_idx = dataload.load_train(use_chached=True,filepath='train_mat_smaller.hdf5',crop_rows=400,crop_cols=400,no=3777,use_heatmap=True)

current_img = np.array(dataset[no])
current_img = current_img.astype('uint8')

# Per fish: get heatmap for direction, idx for position
heatmap_orig,max_idx, prob_fish = heatmap(current_img)
print(ids[no][0].decode('UTF-8'))
yoo = load_single_img(ids[no][0].decode('UTF-8'),convert_bgr=True,transpose=False)
cv2.imshow('works?',yoo)
# Rotate?

plt.imshow([[1,0],[0,1]])
plt.show()