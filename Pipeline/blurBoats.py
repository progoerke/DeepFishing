
import numpy as np
import glob
import sys
import os
from sklearn import cluster
from scipy.misc import imread
import cv2
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage import io

def boat_clusters(filenames, use_ssim=False, min_samples=1, eps=0.7):

    # Read images
    imgs=[]
    imgs_color=[]
    for f in file_names:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img_color=cv2.imread(f)
        imgs.append(img)
        imgs_color.append(img_color)
    print('Number of images {}'.format(len(imgs)))

    # Group images by size.
    imgs_sizes = np.zeros((len(imgs), 1))
    for i, img in enumerate(imgs):
        imgs_sizes[i] = img.size
        sizes_list=np.unique(imgs_sizes)

    ymax_all_sizes=0
    y_all_sizes=np.zeros(len(imgs),dtype=int)

    for isize, img_size in enumerate(sizes_list):

        imgs_list = []
        imgs_indx=[]
        for i in range(len(imgs)):
            if imgs_sizes[i]==img_size:
                #imgs_array[i,:,:] = np.squeeze(img_to_array(imgs[i]))
                imgs_list.append(np.squeeze(img_to_array(imgs[i])))
                imgs_indx.append(i)
        imgs_array = np.asarray(imgs_list)

        # Normalize images to scale (-1, 1)
        mean_imgs = np.mean(imgs_array, axis=0)
        std_imgs = np.std(imgs_array, axis=0)
        imgs_array=(imgs_array-mean_imgs)/(std_imgs+1)

        # Calculate distances between images
        # Use slower structural similarity index, else: faster average absolute difference
        distances = np.zeros((imgs_array.shape[0], imgs_array.shape[0]))
        for i in range(imgs_array.shape[0]):
            print("image #", i)
            if use_ssim:
                for j in range(imgs_array.shape[0]):
                    dist=1-ssim(imgs_array[i], imgs_array[j])
                    distances[i, j] = dist
            else:
                dist = np.mean(np.mean(np.abs(imgs_array[i]-imgs_array), axis=2),axis=1)
                distances[i, :] = dist

        #plt.imshow(distances, interpolation='nearest', cmap=plt.cm.Blues)

        # Cluster images based on distances
        cls = cluster.DBSCAN(metric='precomputed', min_samples=min_samples, eps=eps)
        y = cls.fit_predict(distances)
        print(y)
        for i in range(y.shape[0]):
            if y[i] >=0:
                y_all_sizes[imgs_indx[i]] = y[i] + ymax_all_sizes + 1
            else:
                y_all_sizes[imgs_indx[i]] = 0
        ymax_all_sizes = np.max(y_all_sizes)

    return y_all_sizes, imgs_color

def average_clusters(y, imgs_color):

    # Calculate average image for each cluster
    imgs_averages = [None]*(np.max(y)+1) # np.zeros((np.max(y)+1, img_size[0], img_size[1], 3))
    for icluster in range(np.max(y)+1):
        #directory='./data/train/Boat_clusters/cluster_'+str(min(20,icluster))
        #if not os.path.exists(directory):
        #    os.makedirs(directory)
        for i in range(len(imgs_color)):
            img = imgs_color[i]
            if y[i] == icluster:
                ni = np.sum(y==icluster)
                if imgs_averages[icluster] == None:
                    imgs_averages[icluster] = img/ni
                elif icluster>0:
                    imgs_averages[icluster] += img/ni
        #        io.imsave('./data/train/Boat_clusters/cluster_' + str(min(20, icluster)) + '/' + f[-13:],img)

    return imgs_averages

def blurBoat(icluster, img, imgs_averages, maxblur=1.0):

    img_bw=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diff_array=np.abs(img_bw - cv2.cvtColor(np.squeeze(imgs_averages[icluster]), cv2.COLOR_BGR2GRAY))
    mask_bw = np.squeeze(diff_array) > 30 # mask based on difference with average
    mask=np.zeros((mask_bw.shape[0], mask_bw.shape[1], 3))
    for ich in range(3):
        mask[:,:,ich]=mask_bw
    if 1-np.sum(mask)/mask.size < maxblur:
        mask = cv2.GaussianBlur(mask.astype(np.float), (101,101), 0.) # blurred mask
        blur_img = cv2.GaussianBlur(img, (301, 301), 0.)
        mask_array=np.multiply(mask,img)
        blur_array=np.multiply(1-mask,blur_img)
        maskblur_array=np.add(mask_array,blur_array)
        maskblur_img = np.squeeze(np.uint8(maskblur_array))
    else:
        maskblur_img = img

    return maskblur_img


