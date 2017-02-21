from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
from  os import listdir
import pickle

def load_img_size_number(s1,s2,no):

    dict = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']

    dir = "../../data/small_train"               #location of 'train'
    subdirs = listdir(dir)[1::]
    print(subdirs)

    X = np.zeros((no,s1,s2,3))
    Y = np.zeros((no,8))
    Z = np.zeros((no,10)) #picture size category             
    size_dict = []  #dict for sizes.

    running_idx = 0
    for i,d in enumerate(dict):                 #parse all subdirections
        #print(d)
        sys.stdout.write(".")
        
        files = listdir(dir+"/"+d)              #get all files

        for j, f in enumerate(files):           #parse through all files
            #print(f)
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
                
                #print(running_idx, img.shape)
                #current_img = np.zeros((1,s1,s2,3))  
                #current_img[0,0:img.shape[0],0:img.shape[1],:]=img
                
                # Get from heatmap/box
                center_row = 100
                center_col = 100
                start_crop_row = int(center_row - s1/2)
                stop_crop_row = int(start_crop_row + s1)
                start_crop_col = int(center_row - s2/2)
                stop_crop_col = int(start_crop_col + s2)

                current_img = img[start_crop_row:stop_crop_row,start_crop_col:stop_crop_col,:]

                # alter current_img
                # our code here:
                #...

                X[running_idx,:,:,:] = current_img;     #add to file; beyond frame = zeros...
                Y[running_idx,i]=1                          #store target category
                running_idx += 1

    #pictures
    # pickle.dump(X,open('X_mat.pkl','wb'))
    # pickle.dump(Y,open('Y_mat.pkl','wb'))
    # sys.stdout.write('\n Done :)')
    
    file = h5py.File("X_mat.h5py",'w')
    # Can this work dynamically, I mean the size?
    file.create_dataset("X",data=X)
    file.close()
    #picture category
    file_y = h5py.File("Y_mat.h5py",'w')
    # Can this work dynamically, I mean the size?
    Y_set = file_y.create_dataset('Y',data=Y)
    file_y.close()
    #print(size_dict)
    sys.stdout.write('\n Also hdf5?!')

#load_img_size_number(100,100,16) #smaller_train pkl: 12,2MB h5py: 3,8MB
#load_img_size_number(100,100,40) #small_train pkl: 30,3MB h5py: 9,6MB
load_img_size_number(100,100,3777) #train pkl: --- h5py: 906,5MB