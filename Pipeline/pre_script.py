import pre_network
import dataloader
from skimage import io

p = pre_network.Pre_Network()
#p.train_model(epochs=10)
p.load_model()
data, _, _, _ = dataloader.load_train(filepath='data/train.hdf5',use_cached=True)
patches = p.predict(data[:10])

for i,patch in enumerate(patches):
  io.imsave('test_patches/patch{}.jpg'.format(i),patch)
