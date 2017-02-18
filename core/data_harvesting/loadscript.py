from __future__ import division
import h5py
import numpy as np
import matplotlib.pylab as plt
from  os import listdir

dict = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']

dir = "F:/mlip/train"               #location of 'train'
subdirs = listdir(dir)[1::]

file = h5py.File("X_mat.h5py",'w')
X_set = file.create_dataset("X",(3777,974,1518,3),chunks = (1,974,1518,3),dtype = 'i8')
Y = np.zeros((3777,8)) #picture category     
Z = np.zeros((3777,10)) #picture size category             
size_dict = []  #dict for sizes.

for i,d in enumerate(dict):                 #parse all subdirections
    files = listdir(dir+"/"+d)              #get all files

    for j, f in enumerate(files):           #parse through all files
        img = plt.imread(dir+"/"+d+"/"+f)   #load img

        prt = True                          #check if img has new size
        for k, s in enumerate(size_dict):            
            if img.shape == s:
                prt=False                
                Y[i+j,k]=1                
        if prt:
            size_dict.append(img.shape)
            Y[i+j,len(size_dict)]=1
            
        
        print(i+j, img.shape)
        current_img = np.zeros((1,974,1518,3))  
        current_img[0,0:img.shape[0],0:img.shape[1],:]=img
        X_set[i+j,:,:,:] = current_img;     #add to file; beyond frame = zeros...
        Y[i+j,i]=1                          #store target category

        