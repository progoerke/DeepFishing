from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
from  os import listdir

dict = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']

dir = "../../data/small_train"               #location of 'train'
subdirs = listdir(dir)[1::]
print(subdirs)

file = h5py.File("X_mat.h5py",'w')
X_set = file.create_dataset("X",(3777,974,1518,3),chunks = (1,974,1518,3),dtype = 'f')
#picture category
file_y = h5py.File("Y_mat.h5py",'w')
Y_set = file_y.create_dataset('Y',(3777,8),chunks=(1,8),dtype='i8')
Z = np.zeros((3777,10)) #picture size category             
size_dict = []  #dict for sizes.

running_idx = 0
for i,d in enumerate(dict):                 #parse all subdirections
    print(d)
    files = listdir(dir+"/"+d)              #get all files

    for j, f in enumerate(files):           #parse through all files
        print(f)
        if not(f == '.DS_Store'):
            img = plt.imread(dir+"/"+d+"/"+f)   #load img

            prt = True                          #check if img has new size
            for k, s in enumerate(size_dict):            
                if img.shape == s:
                    prt=False                
                    Z[running_idx,k]=1                
            if prt:
                size_dict.append(img.shape)
                Z[running_idx,len(size_dict)]=1
            
            print(running_idx, img.shape)
            current_img = np.zeros((1,974,1518,3))  
            current_img[0,0:img.shape[0],0:img.shape[1],:]=img
            X_set[running_idx,:,:,:] = current_img;     #add to file; beyond frame = zeros...
            Y_set[running_idx,i]=1                          #store target category
            running_idx += 1
        