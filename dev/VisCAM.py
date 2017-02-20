import cv2
import numpy as np
import os

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.utils.vggnet import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from vis.visualization import visualize_cam
from skimage import io

from VisCAM_NNparams import synset_to_dfs_ids
#from VisCAM_NNparams import detect_configuration

# Build the VGG19 network with ImageNet weights
model = VGG19(weights='imagenet', include_top=True)
#model = InceptionV3(weights='imagenet', include_top=True)
print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

#model_type, index = detect_configuration(model)
#s = "n02512053"  # Imagenet code for "fish"
#s = "n02545841"  # Imagenet code for "Opah fish"
#s = "n02626762"  # Imagenet code for "tuna"
s = "n02536864"  # Imagenet code for "salmon" = 391
pred_class = synset_to_dfs_ids(s)

def get_file_list(path,ext='',queue=''):
    if ext != '': return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]
    else: return [os.path.join(path,f) for f in os.listdir(path)]

# set the path to the folder with the data you downloaded from SURFDrive
data_folder = '../data/'

# timages
image_dir = os.path.join(data_folder, 'train' , 'BET')
image_paths = get_file_list(image_dir, 'jpg')[0] # JPEG files

prob_imgs=[]
heatmaps=[]
for path in image_paths:
    # Predict the corresponding class for use in `visualize_saliency`.
    seed_img = utils.load_img(path, target_size=(224, 224))
    seed_array=np.array([img_to_array(seed_img)])
    probs=model.predict(seed_array)
    prob_imgs.append(probs[0,pred_class])

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_cam(model, layer_idx, pred_class, seed_img, text=utils.get_imagenet_label(pred_class))
    path_hm=path.replace('/train/', '/train/hm_')
    io.imsave(path_hm,heatmap)
    heatmaps.append(heatmap)

np.savetxt(path_hm,prob_imgs)
#cv2.imshow("Original image", seed_img)
cv2.imshow("Class activation heat maps", utils.stitch_images(heatmaps, cols=15))
cv2.waitKey(0)