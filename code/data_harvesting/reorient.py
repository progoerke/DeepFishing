import numpy as np
import skimage.measure
from scipy.stats import multivariate_normal
from scipy import signal
import cv2
import matplotlib.pyplot as plt

import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
#sys.path.append('F:/mlip/DeepFishing-master/DeepFishing-master/code/data_harvesting')


from imgloader import load_single_img
from heatmap_VisCAM import heatmap

'''For this script to work properly, i had to alter the VIScam script to return heatmaps 
in greyscale (makes much more sense procedurally as well...) and delete the square max 
indicator in the centre of mass.'''

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

def orient_fish(img,tx,ty):
    '''reorientate a fish in the image so that its head points to the upper left
    corner. 
    
    @Args:
    -img: The original image ([x,y,3] numpy matrix)
    -xt, yt: Target pixel sizes of the output image.
    
    P.S.: 
    Or top right corner :o(
    ...To be fixed later...
    '''
    tx =int(tx/2)   #using 1/2 of the eventual image size is easier in the end...
    ty =int(ty/2) 
    
    t1x = int(tx*np.sqrt(2))        #leave potential space for the rotation
    t1y = int(ty*np.sqrt(2))
    
    h,m,p,ph = heatmap(img)                 #first heatmap to locate fish
    m=np.array(m)
    
    m0 = m      #original centre
    
    m[0] = min(max(t1y,m[0]),img.shape[0]-t1y)      #find cropping points
    m[1] = (min(max(t1x,m[1]),img.shape[1]-t1x))
    
    m = m.astype(np.int32)
    
    m1 = m      #centre after cropping
    
    img_part = img[(m[0]-ty):(m[0]+ty),(m[1]-tx):(m[1]+tx),:]   #~crop around fish
    
    h2,m2,p2,ph2 = heatmap(img_part)            #second heatmap to inspect fish    
    #ph2 = np.sum(ph2,2)
    
    #cv2.imshow('current',img)
    #cv2.imshow('heat0',ph)
     
    #cv2.imshow('cropped',img_part)
    #cv2.imshow('second heat',ph2)
    
    centroid,theta, covmat = decomp(ph2.astype(np.double))    #decompose heatmap, get rotation angle
    print(theta)    
    rot_mat = cv2.getRotationMatrix2D(tuple([m2[1],m2[0]]),np.rad2deg(theta)-45,1)
    rot_mat1 = cv2.getRotationMatrix2D(tuple([m[1],m[0]]),np.rad2deg(theta)-45,1)
    
    heatmap_rotated = cv2.warpAffine(ph2, rot_mat, tuple([t1y,t1x]),flags=cv2.INTER_LINEAR)    
    image_rotated = cv2.warpAffine(img, rot_mat1, dsize = tuple([img.shape[1],img.shape[0]]),flags=cv2.INTER_LINEAR)
    
    cv2.imshow('rot',image_rotated)
    
    
    
    image_cropped = image_rotated[(m1[0]-ty):(m1[0]+ty),(m1[1]-tx):(m1[1]+tx),:]
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
        
    mag_factor = 3000/np.sum(covmat)
    m1 = (np.multiply(m1,mag_factor).astype(np.int))
    #ty = ty*mag_factor
    #tx = tx*mag_factor
    
    image_rotated = cv2.resize(image_rotated, tuple(np.multiply(image_rotated.shape[0:2], mag_factor).astype(np.int)) , interpolation =cv2.INTER_LINEAR) 
    image_cropped1 = image_rotated[(m1[0]-ty):(m1[0]+ty),(m1[1]-tx):(m1[1]+tx),:]    
    cv2.imshow('rotcrop',image_cropped)
    cv2.imshow('heatcrop',heatmap_cropped)

    return heatmap_rotated,image_rotated, heatmap_cropped, image_cropped


img = load_single_img("../../../../train/BET/img_05262.jpg",convert_bgr=True)
hr,ir, hc, ic = orient_fish(img,350,350)
# tx = 300
# ty = 300
# 
# tx =int(tx/2)   #using 1/2 of the eventual image size is easier in the end...
# ty =int(ty/2) 
# 
# t1x = int(tx*np.sqrt(2))        #leave potential space for the rotation
# t1y = int(ty*np.sqrt(2))
# 
# h,m,p,ph = heatmap(img)                 #first heatmap to locate fish
# m=np.array(m)
# 
# m0 = m      #original centre
# 
# m[0] = min(max(t1y,m[0]),img.shape[0]-t1y)      #find cropping points
# m[1] = (min(max(t1x,m[1]),img.shape[1]-t1x))
# 
# m = m.astype(np.int32)
# 
# m1 = m      #centre after cropping
# 
# img_part = img[(m[0]-ty):(m[0]+ty),(m[1]-tx):(m[1]+tx),:]   #~crop around fish
# 
# h2,m2,p2,ph2 = heatmap(img_part)            #second heatmap to inspect fish    
# #ph2 = np.sum(ph2,2)
# 
# cv2.imshow('current',img)
# cv2.imshow('heat0',ph)
# 
# cv2.imshow('cropped',img_part)
# cv2.imshow('second heat',ph2)
# 
# centroid,theta = decomp(ph2.astype(np.double))    #decompose heatmap, get rotation matrix    
# rot_mat = cv2.getRotationMatrix2D(tuple([m2[1],m2[0]]),np.rad2deg(theta)+45,1)
# rot_mat1 = cv2.getRotationMatrix2D(tuple([m[1],m[0]]),np.rad2deg(theta)+45,1)
# 
# heatmap_rotated = cv2.warpAffine(ph2, rot_mat, tuple([tx,ty]),flags=cv2.INTER_LINEAR)    
# image_rotated = cv2.warpAffine(img, rot_mat1, dsize = tuple([img.shape[1],img.shape[0]]),flags=cv2.INTER_LINEAR)
# 
# cv2.imshow('rot',image_rotated)
# 
# 
# heatmap_cropped = heatmap_rotated #[(m1[0]-ty):(m1[0]+ty),(m1[1]-tx):(m1[1]+tx),:]
# image_cropped = image_rotated[(m1[0]-ty):(m1[0]+ty),(m1[1]-tx):(m1[1]+tx),:]
# cv2.imshow('rotcrop',image_cropped)
# cv2.imshow('rotcrop',heatmap_cropped)

# 
# plt.subplot(1,2,1)
# 
# plt.imshow((current_img))
# 
# 
# plt.subplot(1,2,2)
# 
# plt.imshow(ir)
# plt.show()



