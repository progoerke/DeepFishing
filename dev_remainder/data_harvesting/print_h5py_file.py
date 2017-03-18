import h5py
import sys
import numpy as np
import matplotlib.pylab as plt
import cv2

def print_hdf5_file_structure(file_name) :
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') # open read-only
    item = file #["/Configure:0000/Run:0000"]
    print_hdf5_item_structure(item)
    file.close()
 
def print_hdf5_item_structure(g, offset='    ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if   isinstance(g,h5py.File) :
        print(g.file, '(File)', g.name)
 
    elif isinstance(g,h5py.Dataset) :
        print('(Dataset)', g.name, '    len =', g.shape) #, g.dtype
 
    elif isinstance(g,h5py.Group) :
        print('(Group)', g.name)
 
    else :
        print('WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name)
        sys.exit ( "EXECUTION IS TERMINATED" )
 
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in dict(g).items() :
            subg = val
            print(offset, key)#,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')

no = 1
 
hdf5_file_name = 'train_mat_smaller.hdf5'
print_hdf5_file_structure(hdf5_file_name)
file    = h5py.File(hdf5_file_name, 'r')   # 'r' means that hdf5 file is open in read-only mode
dataset = file['images']
img = np.array(dataset[no])
cv2.imshow('yoo',img.astype(np.uint8))
plt.imshow(img.astype(np.uint8))
plt.show()
sys.exit ( "End of test" )