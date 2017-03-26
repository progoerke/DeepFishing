
import numpy as np
from scipy.signal import medfilt, convolve2d
from scipy.ndimage.filters import gaussian_filter
import scipy.stats as st


kernlen=7 
nsig=1
interval = (2*nsig+1.)/(kernlen)
x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
kern1d = np.diff(st.norm.cdf(x))
kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
kernel = kernel_raw/kernel_raw.sum()


edg =3
d = np.random.random((30,30))
thresh = .5
res = 1

if d.any: #for the case of non zero raw image
    
    d = medfilt(d,[3,3]);
    
    # apply threshold
    if (d.dtype == 'uint8'):
        d=d*np.array(d>thresh).astype(np.uint8)
    else:
        d=d*np.array(d>thresh).astype(np.uint16)

    
    if d.any: #for the case of the image is still non zero
        
        # smooth image
        d=convolve2d(d.astype(np.float32),kernel,'same') 
        
        # Apply again threshold (and change if needed according to SNR)
        d=d*(d>0.9*thresh)
        
        if res: # switch between local maxima and sub-pixel methods
            
            #case 1 # peak find - using the local maxima approach - 1 pixel resolution
                
                # d will be noisy on the edges, and also local maxima looks
                # for nearest neighbors so edge must be at least 1. We'll skip 'edge' pixels.
            sd=d.shape
            x, y=np.where(d[edg:sd[0]-edg,edg:sd[1]-edg]);
            
            # initialize outputs
            cent=[];#
            cent_map=np.zeros((sd));
            
            x=x+edg-1;
            y=y+edg-1;
            for j in range(0,len(y)-1):
                if  (d[x[j],y[j]] == np.max(np.max(d[x[j]-1:x[j]+1,y[j]-1:y[j]+1]))):

                        
                    cent.append([y[j] , x[j]]);
                    cent_map[x[j],y[j]]=cent_map[x[j],y[j]]+1; # if a binary matrix output is desired

        
    else: # in case image after threshold is all zeros
        cent=[];
        cent_map=zeros(size(d));


    
else: # in case raw image is all zeros (dead event)
    cent=[];
