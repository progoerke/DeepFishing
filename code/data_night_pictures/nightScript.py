import numpy as np
import glob
import random
import matplotlib.pyplot as plt
import os
from sklearn import cluster
from sklearn import neighbors
from scipy.misc import imread, imsave


# Devides the training set into two groups depending on the RGB values of the pictures. It uses Knn to create those two
# groups.
#
# Modified version from https://www.kaggle.com/qiubit/the-nature-conservancy-fisheries-monitoring/detecting-night-photos/discussion

# one cluster will be day photos, the other one night photos
knn_cls = 2
# increase this number while training locally for better results
training_imgs = 300

training_files = sorted(glob.glob('./train/*/*.jpg'), key=lambda x: random.random())[:training_imgs]
training = np.array([imread(img) for img in training_files])
training_means = np.array([np.mean(img, axis=(0, 1)) for img in training])
training_features = np.zeros((training_imgs, 3))
for i in range(training_imgs):
    training_features[i][0] = (training_means[i][0] - training_means[i][1])
    training_features[i][0] += (training_means[i][0] - training_means[i][2])
    training_features[i][1] = (training_means[i][1] - training_means[i][0])
    training_features[i][1] += (training_means[i][1] - training_means[i][2])
    training_features[i][2] = (training_means[i][2] - training_means[i][0])
    training_features[i][2] += (training_means[i][2] - training_means[i][1])

kmeans = cluster.KMeans(n_clusters=knn_cls).fit(training_features)
print(np.bincount(kmeans.labels_))


batch = 100

# now load all training examples and cluster them
CLUSTER_FOLDER = os.path.abspath('./train/clustered')
training_filenames = sorted(glob.glob('./train/*/*.jpg'))

cl1 = [[],[],[],[],[],[],[],[]]

# make directories if they doesn't exist
if not os.path.isdir(CLUSTER_FOLDER):
    os.makedirs(CLUSTER_FOLDER)

for cluster_num in range(knn_cls):
    single_cluster_folder = os.path.join(CLUSTER_FOLDER, str(cluster_num))
    if not os.path.isdir(single_cluster_folder):
        os.mkdir(single_cluster_folder)

saved_files = 0
while saved_files < len(training_filenames):
    training_files = training_filenames[saved_files:saved_files+batch]
    training = np.array([imread(img) for img in training_files])
    training_means = np.array([np.mean(img, axis=(0, 1)) for img in training])
    training_features = np.zeros((batch, 3)) #training_imgs    len(training_filenames)
    for i in range(len(training)):
        training_features[i][0] = (training_means[i][0] - training_means[i][1])
        training_features[i][0] += (training_means[i][0] - training_means[i][2])
        training_features[i][1] = (training_means[i][1] - training_means[i][0])
        training_features[i][1] += (training_means[i][1] - training_means[i][2])
        training_features[i][2] = (training_means[i][2] - training_means[i][0])
        training_features[i][2] += (training_means[i][2] - training_means[i][1])

    img_cls = kmeans.predict(training_features)

    for i, img in enumerate(training):
        cluster = img_cls[i]
        save_path = os.path.join(CLUSTER_FOLDER, str(cluster))
        class_name = os.path.basename(os.path.dirname(training_files[i]))
        if class_name == 'ALB':
            cl1[0].append(cluster)
        elif class_name == 'BET':
            cl1[1].append(cluster)
        elif class_name == 'DOL':
            cl1[2].append(cluster)
        elif class_name == 'LAG':
            cl1[3].append(cluster)
        elif class_name == 'NoF':
            cl1[4].append(cluster)
        elif class_name == 'OTHER':
            cl1[5].append(cluster)
        elif class_name == 'SHARK':
            cl1[6].append(cluster)
        elif class_name == 'YFT':
            cl1[7].append(cluster)
        save_path = os.path.join(save_path, class_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(training_files[i]))
        # print save_path
        imsave(save_path, img)
        saved_files += 1

    print(str(saved_files) + "/" + str(len(training_filenames)))


with open("dayNight.txt", "w") as f:
    for i in range(0,len(cl1)):
        for p in cl1[i]:
            f.write(str(p) + ",")
        f.write("\n")
