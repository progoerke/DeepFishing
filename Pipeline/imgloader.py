from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
from  os import listdir
import pickle
import os
import glob
import math
import time

from scipy.misc import imread, imresize

def load_single_img(path,type='int',convert_bgr=False,transpose=False):
    current_img = imread(path)
    if convert_bgr:
        current_img = current_img[:, :, ::-1] # convert to bgr
    if transpose:
        current_img = current_img.transpose((2, 0, 1)) #have color channel as first matrix dim            
    if type is 'float':  
    	current_img = current_img.astype('float32')
    	current_img /= 255
    else:
    	current_img = current_img.astype('uint8')
    return current_img