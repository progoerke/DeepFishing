import h5py
import sys
import numpy as np
import matplotlib.pylab as plt

def print_hdf5_file_structure(file_name) :
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') # open read-only
    item = file #["/Configure:0000/Run:0000"]
    print_hdf5_item_structure(item)
    file.close()
 
def print_hdf5_item_structure(g, offset='    ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if   isinstance(g,h5py.File) :
        print g.file, '(File)', g.name
 
    elif isinstance(g,h5py.Dataset) :
        print '(Dataset)', g.name, '    len =', g.shape #, g.dtype
 
    elif isinstance(g,h5py.Group) :
        print '(Group)', g.name
 
    else :
        print 'WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name
        sys.exit ( "EXECUTION IS TERMINATED" )
 
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in dict(g).iteritems() :
            subg = val
            print offset, key, #,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')

no = 3

hdf5_file_name = 'Y_mat.h5py'
print_hdf5_file_structure(hdf5_file_name)
file    = h5py.File(hdf5_file_name, 'r')   # 'r' means that hdf5 file is open in read-only mode
dataset = file['Y']
print(dataset[no])
 
hdf5_file_name = 'X_mat.h5py'
print_hdf5_file_structure(hdf5_file_name)
file    = h5py.File(hdf5_file_name, 'r')   # 'r' means that hdf5 file is open in read-only mode
dataset = file['X']
img = np.array(dataset[no])
plt.imshow(img.astype(float))
plt.show()
sys.exit ( "End of test" )