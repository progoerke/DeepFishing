import cv2
import numpy as np
import os

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model

from keras.preprocessing.image import img_to_array
from vis.utils import utils
from vis.utils.vggnet import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.utils.np_utils import to_categorical
#from vis.visualization import visualize_cam
from visualization import visualize_cam # Local copy of visualize_cap.py that returns and separate heatmap and location of maximum
from skimage import io

from VisCAM_NNparams import synset_to_dfs_ids
#from VisCAM_NNparams import detect_configuration

# Build the VGG19 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)
#model = VGG19(weights='imagenet', include_top=True)
#model = ResNet50(weights='imagenet', include_top=True)
#model = InceptionV3(weights='imagenet', include_top=True)
#model = Xception(weights='imagenet', include_top=True)

print('Model loaded.')

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'predictions' # VGG16, VGG19, InceptionV3
#layer_name = 'fc1000' # ResNet50
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

#model_type, index = detect_configuration(model)
#s = "n02512053"  # Imagenet code for "fish"
#s = "n02545841"  # Imagenet code for "Opah fish"
#s = "n02626762"  # Imagenet code for "tuna"
s = "n02536864"  # Imagenet code for "silver salmon" = 391
pred_class = synset_to_dfs_ids(s)

def get_file_list(path,ext='',queue=''):
    if ext != '': return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')],  [f for f in os.listdir(path) if f.endswith(''+queue+'.'+ext+'')]
    else: return [os.path.join(path,f) for f in os.listdir(path)]

# set the path to the folder with the data you downloaded from SURFDrive
data_folder = '../data/'

# timages
image_paths=[]
image_folders=['BET','ALB','YFT','DOL','SHARK','OTHER','NoF']
for img_fld in image_folders:
    image_dir = os.path.join(data_folder, 'train' , img_fld)
    if not os.path.isdir(image_dir.replace('/train/', '/train/hm_')):
        os.makedirs(image_dir.replace('/train/', '/train/hm_'))
    image_paths.extend(get_file_list(image_dir, 'jpg')[0]) # JPEG files


prob_imgs=[]
heatmaps=[]
for path in image_paths:
    # Predict the corresponding class for use in `visualize_saliency`.
    orig_img = utils.load_img(path)
    orig_array = np.array([img_to_array(orig_img)])
    orig_size=orig_array.shape
    target_size=(224,224) # VGG16, VGG19, ResNet50
    seed_img = utils.load_img(path, target_size=target_size)
    #seed_img = utils.load_img(path, target_size=(299, 299)) # Inception_V3, Xception
    #x = image.img_to_array(seed_img)
    #x = np.expand_dims(x, axis=0)
    #seed_array = preprocess_input(x)
    seed_array = np.array([img_to_array(seed_img)])
    probs=model.predict(seed_array)
    prob_imgs.append(probs[0,pred_class])

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    # heatmap_overlay : heatmap overlayed on the resized image
    # heatmap : heatmap of resized image
    # max_idx : location of maximum in heatmap of resized image
    heatmap_overlay, heatmap, max_idx = visualize_cam(model, layer_idx, pred_class, seed_img, text=utils.get_imagenet_label(pred_class))

    # The maps are rescaled back to the original image sizes
    # heatmap_overlay_orig : heatmap overlayed on the original image
    # heatmap_orig : heatmap of original image
    # max_orig : location of maximum in heatmap of original image
    heatmap_overlay_orig = cv2.resize(heatmap_overlay, (orig_size[2], orig_size[1]))
    heatmap_orig = cv2.resize(heatmap, (orig_size[2], orig_size[1]))
    max_orig=np.floor(max_idx[0]*orig_size[1]/target_size[0]), np.floor(max_idx[1]*orig_size[2]/target_size[1])

    # Saving the heatmap overlayed on the original image to disk
    path_hm=path.replace('/train/', '/train/hm_')
    io.imsave(path_hm,heatmap_overlay_orig)
    heatmaps.append(heatmap_orig)
    break

np.savetxt(path_hm,prob_imgs)
#cv2.imshow("Original image", seed_img)
cv2.imshow("Class activation heat maps", utils.stitch_images(heatmaps, cols=15))
cv2.waitKey(0)