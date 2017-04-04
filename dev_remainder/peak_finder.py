
import numpy as np
from scipy.signal import medfilt, convolve2d
from scipy.ndimage.filters import gaussian_filter
import scipy.stats as st
import cv2


def find_peaks(img, thresh = .5, edg = 10,kernel = 50):
    '''
    Finds peak location in a 2d density.
    
    @args:
    -img: [x,y] density
    
    @opt args:
    -thresh: Threshold value (-> fight noise)
    -edg: Edge length of the peak search kernel
    -kernel: Size of a gaussian kernel that blurs the density (-> fight noise)
    
    @output:
    -cent: [[x,x,x],[y,y,y]] values of all the peaks found in the image 
    -cent_map: peak topography on a boolean matrix of [img.shape]
    '''
    kernlen=kernel
    nsig=2
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    
    
    if img.any: #for the case of non zero raw image
        
        #img = medfilt(img,[5,5]);
        #img=convolve2d(img.astype(np.float32),kernel,'same') 
        # apply threshold
        cv2.imshow('o',img)
        
        kernel = kernel/np.max(kernel)
        cv2.imshow('k',kernel)
        
        if (img.dtype == 'uint8'):
            img=np.array(img>thresh).astype(np.uint8)
        else:
            img=np.array(img>thresh).astype(np.uint16)
    
        #img=np.array(img>thresh).astype(np.uint8)
        
        if img.any: #for the case of the image is still non zero
            
            # smooth image
            img=convolve2d(img.astype(np.float32),kernel,'same') 
            cv2.imshow('s',img)
            # Apply again threshold (and change if needed according to SNR)
            #print(img)
            img=img*(img>0.9*thresh)
            #print(img)
            #print(0.9 * thresh)

                
                #peak find - using the local maxima approach - 1 pixel resolution   
                    # d will be noisy on the edges, and also local maxima looks
                    # for nearest neighbors so edge must be at least 1. We'll skip 'edge' pixels.
            sd=img.shape
            [x, y]=np.where(img[edg:sd[0]-edg,edg:sd[1]-edg]);
            #print(x,y)
            # initialize outputs
            cent=[];#
            cent_map=np.zeros((sd));
            
            x=x+edg
            y=y+edg
            for j in range(0,len(y)-1):
                if  (img[x[j],y[j]] == np.max(np.max(img[x[j]-1:x[j]+2,y[j]-1:y[j]+2]))):
                        cent.append([y[j] , x[j]]);                        
                        cent_map[x[j],y[j]]=cent_map[x[j],y[j]]+1; # if a binary matrix output is desired
    
            
        else: # in case image after threshold is all zeros
            cent=[];
            cent_map=zeros(size(img));
    
    
        
    else: # in case raw image is all zeros (dead event)
        cent=[];
    
    return cent, cent_map 

def fish_locator_3000(map,x=400,y=400):    
    '''
    Peak location applied to our task.
    
    @args:
    -map: [x,y] p values of the patches for one single img (-> conserve topographical location).  
    
    @opt args:
    -x: width of the original image (the one the map was obtained from)
    -y: height of the original image 
    
    @output:
    -map1: [[x,x,x],[y,y,y]] values of all the peaks found in the image 
    -cent_map: peak topography on a boolean matrix of [img.shape]
    ''' 
    enlarged = cv2.resize(np.divide(map,np.max(map)),(x,y))
    map = np.divide(map,np.max(map))
    
    map1,cmap = find_peaks(enlarged,.6,5,170)
    
    return map1,cmap

'''
P = np.array([[0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0],
              [0,0,0,0,0,1,0],
              [0,0,0,1,4,2,1],
              [0,2,3,0,0,0,1],
              [0,2,1,0,1,0,0],
              [0,1,0,0,0,0,0]])

P=P/np.max(P)
              
img = load_single_img('F:/mlip/train/ALB/img_00274.jpg',convert_bgr=True)

cv2.imshow('img',img)

pts,arr = fish_locator_3000(P,img.shape[1],img.shape[0])
for i,p in enumerate(pts):
    crpd = crop_around(img,250,250,p[0],p[1])
    cv2.imshow(str(p),crpd[0])
    if i>2: break

              
              '''