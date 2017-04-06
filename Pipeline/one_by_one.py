import dataloader

if __name__ == '__main__':
    ids_class = np.empty()
    data, labels, train_indx, val_indx = dataloader.load_train(use_cached=False, use_heatmap=False)

    for c in range(8):
        ids_class = np.where(labels[:,c] == 1)[0]
        # the conversion to list of the numpy array of indeces is because hdf5 support only lists or boolean arrays for indexing
        #data_single_class = data[ids_class.tolist()]
        #label_single_class = labels[ids_class.tolist()]
        n_val_samples = ids_class * 10 // 100
        val_idx = np.random.choice(ids_class, n_val_samples, replace=False)
        train_idx = np.setdiff1d(ids_class, val_idx)
        
        run_model((params['lr'],64,'sgd'), data=data, labels=labels, train_indx=train_indx, val_indx=val_indx)  

        
