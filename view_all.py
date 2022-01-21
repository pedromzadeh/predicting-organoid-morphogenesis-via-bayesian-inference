import image_processing_tools as ipt
import branching_tools as obt
import load_module as ld
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Read me:
# Takes in an organoid example and shows each frame statically,
# along with its CM. I can use the mouse to pin the branching
# direction and compute the eyeballed branching angle.
# All frames are shown, with background coming from Blue channel.

type = 'No FGF/MEK1DD'
year = '2019'
IDs = ld.load(type=type, year=year)
root = '/windir/c/Users/Pedrom Zadeh/Documents/Research/Data Storage/'
path = root + '{}Data/{}/XY_Pos'.format(year,type)
print(IDs)

for id in IDs:
   # read this id's tif
   # split the channels + frames --> get all background (blue channel) frames
   path_id = path + str(id) + '.tif'
   channels = ipt.split_tif(path_id)
   raw_back = np.array(channels[1::2])

   # get processed background frames
   tvs = np.ones(len(raw_back))*600
   backs_id = map(ipt.get_background,raw_back,tvs)
   backs_id = list(backs_id)
   cms = list(map(ipt.get_CM,backs_id))

   fig_1, axs = plt.subplots(3,3, figsize=(8,8))
   fig_1.suptitle('ID {}'.format(id))

   # visualize all frames for this id at once
   for k in range(len(backs_id)):
      j = k % 3
      i = int(k / 3)
      axs[i,j].imshow(backs_id[k],cmap='Greys')
      axs[i,j].scatter(cms[k][0],cms[k][1],color='red')
      axs[i,j].set_title('CM = [{:.2f},{:.2f}]'.format(cms[k][0],cms[k][1]))
   plt.show()
