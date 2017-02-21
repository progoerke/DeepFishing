from __future__ import division
import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
from  os import listdir
import pickle
import time

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
        sys.stdout.flush()
        
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
                center_row = 250
                center_col = 500
                start_crop_row = int(center_row - s1/2)
                if start_crop_row < 0:
                    start_crop_row = 0
                stop_crop_row = int(start_crop_row + s1)
                if stop_crop_row > img.shape[0]:
                    stop_crop_row = img.shape[0]
                    start_crop_row = stop_crop_row - s1
                start_crop_col = int(center_row - s2/2)
                if start_crop_col < 0:
                    start_crop_col = 0
                stop_crop_col = int(start_crop_col + s2)
                if stop_crop_col > img.shape[1]:
                    stop_crop_col = img.shape[1]
                    start_crop_col = stop_crop_col - s2

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
    # sys.stdout.write('\n Pickle done')
    
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
    sys.stdout.write('\n Doooone :)\n')

start = time.time()
#load_img_size_number(100,100,16) #smaller_train pkl: 12,2MB h5py: 3,8MB
#load_img_size_number(100,100,40) #small_train pkl: 30,3MB h5py: 9,6MB
load_img_size_number(200,200,3777)
end = time.time()
print(end - start)
#100, 100, train pkl: --- h5py: 906,5MB
#200, 200, train pkl: --- h5py: 3,63GB 7,8 sec
#300, 300, train pkl: --- h5py: 8,16GB 18.6 sec
#400, 400, train pkl: --- h5py: 14,5GB 31.3 sec
#500, 500, train pkl: --- h5py: 22,7GB 46.9 sec