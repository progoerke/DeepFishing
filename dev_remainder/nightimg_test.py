from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
from  os import listdir
import pickle
import os
import glob
import math
import time
import cv2
from imgloader import load_single_img


# function to do histogram matching
def get_histogram_matching_lut(one_channel, day_img, n_bins, plot_me=False):
   
    H_input = np.zeros((n_bins))
    one_channel = (one_channel - one_channel.flatten().min())/(one_channel.flatten().max() - one_channel.flatten().min())
    h_input = np.histogram(one_channel, bins=np.linspace(0., 1., n_bins+1))
    H_input = np.cumsum(h_input[0])
        
    ### Creates fake histogram ###  
    H_template = np.zeros((3,n_bins))
    day_img = (day_img - day_img.flatten().min())/(day_img.flatten().max() - day_img.flatten().min())
    h_input = np.histogram(day_img, bins=np.linspace(0., 1., n_bins+1))
    H_template = np.cumsum(h_input[0])
    #### ####  
        
    LUT = np.zeros(len(h_input[0]))
    # Get next best index of the target histogram
    for i in range(len(H_template)):
        input_v = H_input[i]
        new_idx = (np.abs(H_template-input_v)).argmin()
        LUT[i] = new_idx
    # apply histogram matching (all_channels * (n_bins-1)).astype(int)
    equalized_channels = LUT[(one_channel*(n_bins-1)).astype(int)]
    
    equalized_channels = (equalized_channels - equalized_channels.flatten().min())/(equalized_channels.flatten().max() - equalized_channels.flatten().min())

    # plot cumulative histogram
    if plot_me:
        plt.suptitle('CUMULATIVE HISTOGRAMS')
        plt.subplot(1,2,1); plt.hist(one_channel.flatten())
        plt.title('cumulative histogram input')
        plt.subplot(1,2,2); plt.hist(equalized_channels.flatten())
        plt.title('cumulative histogram transformed')
        plt.show()
    
    return equalized_channels

dayimg = load_single_img('../../../../train/ALB/img_00003.jpg',convert_bgr=True)
print(dayimg.shape)
dayimg = np.transpose(dayimg, (2,0,1))
print(dayimg.dtype)
nightimg = load_single_img('../../../../train/ALB/img_00019.jpg',convert_bgr=True)
print(nightimg.shape)
nightimg = np.transpose(nightimg, (2,0,1))
print(nightimg.dtype)

result = np.zeros((720,1280,3))

result[:,:,0] = get_histogram_matching_lut(nightimg[0,:,:], dayimg[0,:,:], 400, plot_me=False)
result[:,:,1] = get_histogram_matching_lut(nightimg[1,:,:], dayimg[1,:,:], 400, plot_me=False)
result[:,:,2] = get_histogram_matching_lut(nightimg[2,:,:], dayimg[2,:,:], 400, plot_me=False)

cv2.imshow('transformed',result)
nightimg = np.transpose(nightimg, (1,2,0))
cv2.imshow('input',nightimg)
plt.imshow(result)
plt.show()