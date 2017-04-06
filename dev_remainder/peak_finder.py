
import numpy as np
from scipy.signal import medfilt, convolve2d
from scipy.ndimage.filters import gaussian_filter
import scipy.stats as st
import cv2
#import fish_utils

def find_peaks(img, thresh = .6, edg = 10,kernel = 75):
    '''
    Finds peak location in a 2d density.
    
    @args:
    -img: [x,y] density
    
    @opt args:
    -thresh: Threshold value (-> to fight noise)
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
        #cv2.imshow('o',img)
        
        kernel = kernel#/np.max(kernel)
        #cv2.imshow('k',kernel)
        
        #if (img.dtype == 'uint8'):
         #   img=img*np.array(img>thresh).astype(np.uint8)
        #else:
            #img=img*np.array(img>thresh).astype(np.uint16)
    
        #img=np.array(img>thresh).astype(np.uint8)
        
        if img.any: #for the case of the image is still non zero
            
            # smooth image
            img=convolve2d(img.astype(np.float32),kernel,'same') 
            #cv2.imshow('s',img)
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
            vals = [];
            cent_map=np.zeros((sd));
            
            x=x+edg
            y=y+edg
            for j in range(0,len(y)-1):
                if  (img[x[j],y[j]] == np.max(np.max(img[x[j]-1:x[j]+2,y[j]-1:y[j]+2]))):
                    vals.append(img[x[j],y[j]])    
                    cent.append([y[j] , x[j]]);                        
                    cent_map[x[j],y[j]]=cent_map[x[j],y[j]]+1; # if a binary matrix output is desired
    
            
        else: # in case image after threshold is all zeros
            cent=[];
            cent_map=zeros(size(img));
    
    
        
    else: # in case raw image is all zeros (dead event)
        cent=[];
    
    return cent,vals,cent_map 

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
    
    map1,vals,cmap = find_peaks(enlarged,.1,5,170)
    
    return map1,vals,cmap

'''
P = np.array([[0,0,0,0,0,0,0],
              [0,1,0,0,0,0,0],
              [0,0,0,0,0,1,0],
              [0,0,0,1,4,2,1],
              [0,2,3,0,0,0,1],
              [0,2,1,0,1,0,0],
              [0,1,0,0,0,0,0]])

P = np.array([[ -3.81675208,  -6.54932308,  -4.97684908,   7.28140444,
         10.12865579,  13.63543689,  14.54952502,  11.38474917,
          3.72880727,   2.60805249,   1.65222178],
       [ -3.04542071,  -3.41754356,  -4.69458894,  10.3975811 ,
         12.46151066,  15.97447348,  15.67757249,  10.10120457,
          9.24266458,   4.20558047,   0.61398578],
       [ -2.03547332,   8.05736423,  10.78018045,   3.5596019 ,
         10.09448874,  15.79542899,  15.05048156,   5.71847127,
          4.89770025,   1.99748778,  -1.39357132],
       [ -4.07049102,  11.20053202,  11.70211697,  -6.3040508 ,
          3.6040355 ,   9.47612023,  10.82564044,   6.34597921,
          2.55502307,   1.02602756,  -4.83040607],
       [ -3.41752014,  -4.36316919,  -4.23513603,  -7.78808963,
         -1.06252003,  11.55272698,  10.95136166,  12.57449293,
          4.63245809,   2.34160814,  -2.97414173],
       [ -0.2512008 ,  -5.26988065,  -4.44280243,  -5.92587066,
         -0.3956427 ,  14.48501134,  15.29126549,  16.02579141,
          0.99494966,  -1.0883472 ,   0.28443252]])

P=P+np.min(P)
P=P/np.max(P)
              
img = load_single_img('F:/mlip/train/ALB/img_00055.jpg',convert_bgr=True)

cv2.imshow('img',img)

#pts,vals,arr = fish_locator_3000(P,img.shape[1],img.shape[0])
for i,p in enumerate(pts):
    img1 = zoom_around(img,p[1],p[0],1.2/np.exp(vals[i]))
    crpd = crop_around(img1,250,250,p[0],p[1])
    cv2.imshow(str(p),crpd[0])
    if i>5: break
'''
