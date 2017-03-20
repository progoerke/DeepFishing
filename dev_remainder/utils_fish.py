'''@author sebastiantiesmeyer

A couple handy functions for our noble cause.
'''
#some tax-free imports!
import cv2
import numpy as np
from copy import deepcopy

def crop_around(img, x , y, centre_x= None ,centre_y= None):
    '''Crop image around [centre_x, centre_y] and return new image of shape [x,y]
    note: x and y are image coordinates, not matrix coordinates, so y=height, x=width. 
    
    @args:
    -img: [m,n,3] image file
    -x,y: New image shape

    @opt args:
    -centre_x, centre_y: Transformation centre. If None, image centre is used.
        
    returns transformed image, new centre coords
     '''
    tx =int(x/2)   #using 1/2 of the eventual image size is easier in the end...
    ty =int(y/2) 
    
    if centre_x is None: centre_x = int(img.shape[1]/2)
    if centre_y is None: centre_y = int(img.shape[0]/2)    
    
    centre_y = deepcopy(int(min(max(ty,centre_y),img.shape[0]-ty)))      #find cropping points
    centre_x = deepcopy(int(min(max(tx,centre_x),img.shape[1]-tx)))
    
    img = img[(centre_y-ty):(centre_y+ty),(centre_x-tx):(centre_x+tx),:]    
    centre_x = int(img.shape[1]/2)
    centre_y = int(img.shape[0]/2)
    return img, centre_x, centre_y 
    

        
def scale_fish(img, x, y, centre_x = None, centre_y = None):
    '''
    rescale image to given pixel sizes.
    note: x and y are image pixel sizes, not matrix coordinates, so y=height, x=width.
    
    @args:
    -img: [m,n,3] image file
    -x,y: New image shape
    @opt args:
    -centre point recalculated (no effect on scaling)
    
    returns image, new centre
    '''
    if centre_x is not None:
        centre_x = int(centre_x * x/img.shape[1])
        centre_y = int(centre_y * y/img.shape[0])
               
    return cv2.resize(img, (x,y),interpolation =cv2.INTER_LINEAR), centre_x, centre_y


def rotate_fish(img, x,y, theta):
    '''
    rotate image to theta degrees around given coordinates.
    note: x and y are image coordinates, not matrix coordinates, so y=height, x=width.
    
    @args:
    -img: [m,n,3] image file
    -x,y: rotation centre
    -theta: angle (degree)
    
    returns rotated image (same shape as input, edges black.)
    '''
        
    rot_mat = cv2.getRotationMatrix2D((x,y), theta,1)
    return cv2.warpAffine(img, rot_mat, tuple([img.shape[1],img.shape[0]]) ,flags=cv2.INTER_LINEAR) 
    

def shift_fish(img, x,y):
    '''
    shift image content around
    note: x and y are image coordinates, not matrix coordinates, so y=height, x=width.
    
    @args
    -img: [m,n,3] image file
    -x,y: pixels to shift 
    '''
    
    M = np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))  
    
def zoom_around(img, x_centre,y_centre, scope):
    '''
    zoom image
    note: x and y are image coordinates, not matrix coordinates, so y=height, x=width.
    
    @args
    -img: [m,n,3] image file
    -x_centre,y_centre: focal point of the transformation
    -scope: zooming ratio  
    '''    
    M = np.float32([[scope,0,((y_centre-(y_centre*scope)))],[0,scope,((x_centre-(x_centre*scope)))]])   
    return   cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))

def scale_crop(img,target_x, target_y):
    '''crop/scale the image with as little data derogation as possible
    note: x and y are image pixel sizes, not matrix coordinates, so y=height, x=width.
    
    @args
    -img: [m,n,3] image file
    -traget_x,target_y: Shape of the target image
    
    '''
    ari = img.shape[1]/img.shape[0]
    art = target_x/target_y
    
    if (float(ari)/art)<1:
        img1,_,_ = scale_fish(img,target_x,int(int(img.shape[0])*target_x/int(img.shape[1])))
    else:
        img1,_,_ = scale_fish(img,int(int(img.shape[1])*target_y/int(img.shape[0])),target_y)
    
    return crop_around(img1,target_x,target_y)[0]
    
    
    
    