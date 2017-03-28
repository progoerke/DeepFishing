import numpy as np
import skimage.measure
from scipy.stats import multivariate_normal
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import sys
import os
import inspect
from copy import deepcopy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.append('F:/mlip/git/DeepFishing/code/data_harvesting')


from imgloader import load_single_img
from utils_fish import *
from heatmap_VisCAM import heatmap


def decomp(p):
    ''' Fit 2d gaussian to the image histogram,
        return centroid moment & angle. 
        
        @args:
        -n: [x,y] numpy array '''
    p = np.array(p)
    moment = skimage.measure.moments(p,1)
    centroid = [moment[0,1]/moment[0,0],moment[1,0]/moment[0,0]]
    c_moment = skimage.measure.moments_central(p,centroid[0],centroid[1])
    
    mu20 = np.array([c_moment[2,0]/c_moment[0,0]])
    mu02 = np.array([c_moment[0,2]/c_moment[0,0]])
    mu11 = np.array([c_moment[1,1]/c_moment[0,0]])
    cov_mat = np.array([[mu20,mu11],[mu11,mu02]])   #cov(gaussian) fitted to p
    
    theta = 1/2*np.arctan((2*mu11)/(mu20-mu02))     #eigen decomposition -> rotation angle
    
    
    #p1 = cv2.warpAffine(p, rot_mat, p.shape,flags=cv2.INTER_LINEAR)
    return centroid,theta, cov_mat

def orient_fish(img,tx,ty, m = None):
    '''reorientate a fish in the image so that its head points to the upper left
    corner. 
    
    @Args:
    -img: The original image ([x,y,3] numpy matrix)
    -xt, yt: Target pixel sizes of the output image.
    @opt_args
    -m: Heatmap max heat if you have one, otherwise scripts calculates one itself.
    
    P.S.: 
    Or top right corner :o(
    ...To be fixed later...
    '''
    tx =int(tx/2)   #using 1/2 of the eventual image size is easier in the end...
    ty =int(ty/2) 
      
    if m is None:
        _,m,_,p = heatmap(img)                 #first heatmap to locate fish
        m=np.array(m)
    
    m = m.astype(np.int32)
    
    m1 = deepcopy(m)      #centre after cropping

    m3 = deepcopy(m)
    m3[0] = ((img.shape[0]/2) -( m[0]))#+(m2[0]-(ty)))
    m3[1] = ((img.shape[1]/2) -( m[1]))#+(m2[1]-(tx)))
      
    img_centralized = shift_fish(img,m3[1],m3[0])
    #cv2.imshow('cent',img_centralized)
    #cv2.imshow('heatmap',p)
    
    #print(m,m3,img.shape,img_centralized.shape)
    
    m[0] = int(img.shape[0]/2)
    m[1] = int(img.shape[1]/2)
    
    img_part,m1[1],m1[0] = crop_around(img, ty, tx ,m[1],m[0])
    _,m2,_,h2 = heatmap(img_part) #second heatmap to inspect fish  
    
    #cv2.imshow('current',img)
    #cv2.imshow('heat0',ph)
     
    #cv2.imshow('cropped',img_part)
    #cv2.imshow('second heat',ph2)
    
    centroid,theta, covmat = decomp(h2.astype(np.double))    #decompose heatmap, get rotation angle 
      
    image_rotated = rotate_fish(img_centralized, m[1],m[0], np.rad2deg(theta-45))
    heatmap_rotated = rotate_fish(h2, tx, ty, np.rad2deg(theta-45))
    
    cv2.imshow('rot',image_rotated)
  
    image_cropped,_,_ = crop_around(image_rotated,tx*2,ty*2, m[1], m[0])   # [(m1[0]-ty):(m1[0]+ty),(m1[1]-tx):(m1[1]+tx),:]
    #cv2.imshow('crp',image_cropped)
    heatmap_cropped = cv2.resize(heatmap_rotated, (image_cropped.shape[1],image_cropped.shape[0]),interpolation =cv2.INTER_LINEAR) 
    
    #cv2.imshow('heatrot',heatmap_rotated)
    
    #create a filter and run across the image to estimate orientation of the fish:
    
    ftr = np.zeros((40,40,3))
    cv2.ellipse(ftr,(0,0),(45,15),45,0,360,(255,0,0), 2)
    ift = signal.fftconvolve(ftr,image_cropped)
    ift1 = signal.fftconvolve(np.rot90(ftr,2,[0,1]),image_cropped) 
    if(np.sum(ift**2)-np.sum(ift1**2)):
        image_cropped = np.rot90(image_cropped,2,[0,1])
        image_rotated = np.rot90(image_rotated,2,[0,1])
        m3= (img.shape[0]-m1[0],img.shape[1]-m[1])
        heatmap_cropped = np.rot90(heatmap_cropped,2,[0,1])
        
    mag_factor = 5000/sum(covmat[0,0],covmat[1,1])
    m1 = (np.multiply(m1,mag_factor))
    print(mag_factor)
    #ty = ty*mag_factor
    #tx = tx*mag_factor
    
    #image_rotated = cv2.resize(image_rotated, tuple(np.multiply(image_rotated.shape[0:2], mag_factor).astype(np.int)) , interpolation =cv2.INTER_LINEAR) 
    image_zoomed = zoom_around(image_rotated,m3[1],m3[0],mag_factor)  #image_rotated[(m1[0]-ty):(m1[0]+ty),(m1[1]-tx):(m1[1]+tx),:]    
    image_cropped1 = crop_around(image_zoomed,tx*2,ty*2,image_zoomed.shape[1]/2,image_zoomed.shape[0]/2)[0]
    #cv2.imshow('rotcrop',image_cropped)
    #cv2.imshow('rotcrop1',image_cropped1)
    # cv2.imshow('heatcrop',heatmap_cropped)
    # cv2.imshow('orig',img)
    #cv2.imshow('zmd',image_zoomed)

    return heatmap_rotated,image_cropped, heatmap_cropped, image_cropped1


img = load_single_img("../../../../train/BET/img_00883.jpg",convert_bgr=True)
hr,ir, hc, ic = orient_fish(img,350,350)
cv2.imshow('orig',ic)
cv2.imshow('orig',ir)


