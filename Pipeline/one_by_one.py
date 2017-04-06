import dataloader
import numpy as np

if __name__ == '__main__':
		
	data, labels, train_indx, val_indx = dataloader.load_train(filepath='data/train_mat.hdf5',use_cached=False, mode='resize')

	num_classes = 8
	for c in range(num_classes):
		ids_class = np.where(labels[:,c] == 1)[0]
		# the conversion to list of the numpy array of indeces is because hdf5 support only lists or boolean arrays for indexing
		#data_single_class = data[ids_class.tolist()]
		#label_single_class = labels[ids_class.tolist()]
		#n_val_samples = ids_class * 10 // 100
		#val_idx = np.random.choice(ids_class, n_val_samples, replace=False)
		#train_idx = np.setdiff1d(ids_class, val_idx)
		np.random.shuffle(ids_class)
		num_val = int(0.1 * len(ids_class)) 
		val_indx = ids_class[:num_val]
		train_indx = ids_class[num_val:]        

		run_model((0.166,64,'sgd'), data=data, labels=labels, train_indx=train_indx, val_indx=val_indx)  

        
