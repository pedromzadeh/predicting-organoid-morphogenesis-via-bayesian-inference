import image_processing_tools as ipt
import branching_tools as obt
import load_module as ld
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
from json import JSONEncoder
import time
import sys

# Read Me:
#  - loads raw data from experimental tif images
#  - extracts solid background, edges (Image.Filter), and mek for all frames, all examples
#  - overlays all frames w.r.t frame(t=0) CM
#  - pickles in .json overlaid frames for future use
#  - option to visualize data as well for sanity checks

# matplotlib fonts
params = {'font.family' : 'serif', 'font.size' : 15, 'font.weight' : 'bold'}
mpl.rcParams.update(params)

# JSON numpy array encoder
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Load data
type = 'FGF/MEK1DD'
year = '2019'
IDs = ld.load(type=type, year=year)
root = '/windir/c/Users/Pedrom Zadeh/Documents/Research/Data Storage/'
root = root + '{}Data/{}/XY_Pos'.format(year,type)

# holds respective features, all shifted and overlaid to have same CM
edges = []
backs = []
meks = []

t0 = time.time()
for id in IDs:
   print('Extracting and overlaying background, mek, and edges for ID ', id)

   # read this id's tif
   # split the channels + frames --> get all mek and background frames
   # crop to speed up process later on
   # drop last frame
   path = root + str(id) + '.tif'
   channels = ipt.split_tif(path)
   raw_mek, raw_back = np.array(channels[0::2]), np.array(channels[1::2])
   # raw_back = raw_back[:-1,100:1000,100:1000]
   # raw_mek = raw_mek[:-1,100:1000,100:1000]
   raw_back = raw_back[:-1,:,:]
   raw_mek = raw_mek[:-1,:,:]


   # get processed background frames
   tvs = np.ones(len(raw_back))*600
   backs_id = map(ipt.get_background,raw_back)
   backs_id = list(backs_id)

   # get edges from processed background frames
   # not doing this after overlaying backs due to int and uint16 types
   edges_id = map(ipt.get_edges,backs_id)
   edges_id = list(edges_id)

   # get processed mek frames
   tvs = np.ones(len(raw_mek)) * 350
   meks_id = map(ipt.get_MEK1DD,raw_mek,tvs)
   meks_id = list(meks_id)

   # generate empty list to pass as control
   reds_id = np.zeros(len(meks_id))

   # overlay all frames w.r.t first frame's CM calculated from the background
   ref_org = {'back' : backs_id[0], 'mek' : meks_id[0], 'edge' : edges_id[0], 'red' : reds_id[0]}
   org_list = {'backs' : backs_id[1:], 'meks' : meks_id[1:], 'edges' : edges_id[1:], 'reds' : reds_id}
   overlaid_backs_id, overlaid_meks_id, overlaid_edges_id, _ = ipt.overlay_all_frames(org_list,ref_org,control=False)

   # visualize all extracted features
   for fr in range(len(backs_id)):
      fig, axs = plt.subplots(1,4,figsize=(15,5))
      axs[0].set_title('Background')
      axs[0].imshow(overlaid_backs_id[fr],cmap='Reds')
      axs[0].axis('off')
      axs[1].set_title('Edge')
      axs[1].imshow(overlaid_edges_id[fr],cmap='Reds')
      axs[1].axis('off')
      axs[2].set_title('Mek')
      axs[2].imshow(overlaid_meks_id[fr],cmap='Greens')
      axs[2].axis('off')
      axs[3].set_title('Overlaying frames')
      axs[3].imshow(overlaid_meks_id[fr],cmap='Greens',alpha=0.5)
      axs[3].imshow(overlaid_backs_id[fr],cmap='Reds',alpha=0.5)
      axs[3].imshow(overlaid_edges_id[fr],alpha=0.5)
      axs[3].axis('off')
      plt.tight_layout()
      plt.savefig('Visuals/Background-Edges-Mek Extraction/ID_{}_Fr_{}.png'.format(id,fr),dpi=100)
      plt.close()

   edges.append(overlaid_edges_id)
   backs.append(overlaid_backs_id)
   meks.append(overlaid_meks_id)

print('Total time taken to finish: ', time.time()-t0)

print('Pickling background...')
with open('JSON Data/background_FGFMEK.json', 'w') as f:
   json.dump({'backs' : backs},f,cls=NumpyArrayEncoder)
print('Pickling edges...')
with open('JSON Data/edges_FGFMEK.json', 'w') as f:
   json.dump({'edges' : edges},f,cls=NumpyArrayEncoder)
print('Pickling mek...')
with open('JSON Data/meks_FGFMEK.json', 'w') as f:
   json.dump({'meks' : meks},f,cls=NumpyArrayEncoder)
