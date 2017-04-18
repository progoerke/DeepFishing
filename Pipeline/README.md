# How to
To train a model and create a submission file with the predictions on the test set, call `python3 hyper.py -r`. (If you run for the first time, check the installation comments under **Installation**)

# Content
| File | |
|---|---|
| BoatIDs | Folder containing average images of clustered boats
| Classifiers | Folder containing classifiers
| - CCN.py | Convolutional Neural Network
| - VGG_16.py | VGG 16
| - classifier_base.py | Inheriting classifier structure
| - detector.py | Fish detector classifier
| - inceptionV3.py | Inception V3
| - resNet.py | ResNet
| bb_data | Folder containing hand labeled bounding boxes downloaded from the kaggle discussion forum
| VGG_CV.py | Cross validation using VGG16
| VisCAM_NNparams.py | Helper script for heatmaps, modified version, original from keras-vis package
| blurBoats.py | Finding clusters of boat in order to create average image and blur out this average from all instances in cluster
| boat_id_clsfr.py | Classifier based on distribution of fish over boat clusters
| dataloader.py | Loading function for training and test data. Options are using a sliding window with heatmap, a heatmap only once, or simple resizing.
| ensemble.py | Combining different previously trained models to create a common prediction. Options are mean, max, or a network learning to create the prediction.
| fish_utils.py | Utility functions for scaling, cropping, shifting, rotation, etc.
| heatmap_VisCAM.py | Creating a heatmap for fish detection
| hyper.py | Training script for Inception network. Includes data augmentation. Can be called to run one parameter setting or optimize over parameter space. Also creates a submission file.
| hyper_r.py | Training script for ResNet. Specifications same as hyper.py.
| imgloader.py | Loads single image
| jsonloader.py | Load data with bounding box data as masks
| labelBoatIDs.py | Label boats according to boat clusters
| one_by_one.py | Create one classifier per fish class
| pre_network.py | Network trained on detection of fish position from bounding boxes. Output is then fed to other classifiers.
| pre_script.py | Script executing the training of pre_network.py
| save_on_disk.py | Save hdf5 file contents to disk for inspection
| stacker.py | Stacking script training model over individual cross validation split models
| visualization.py | Helper script for heatmaps, modified version, original from keras-vis package

# Installation
For the code several preparations are needed
1. OpenCV 3 installation
2. keras-vis Python package installation from github, e.g. with `pip3 install git+https://github.com/raghakot/keras-vis`
3. Change source code of keras-vis package in `/usr/local/lib/python3.6/site-packages/vis/utils`, line 31:
```
def reverse_enumerate(iterable):
    """Enumerate over an iterable in reverse order while retaining proper indexes, without creating any copies.
    """
    return itertool.izip(reversed(range(len(iterable))), reversed(iterable))
    ```
    change to
    ```
    return zip(reversed(range(len(iterable))), reversed(iterable))
    ```
4. Put training data in Pipeline/data/train/
