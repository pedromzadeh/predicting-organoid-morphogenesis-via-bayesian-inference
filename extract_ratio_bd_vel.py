import image_processing_tools as ipt
import branching_tools as obt
import load_module as ld
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import sys
import os

# Read Me:
#  - loads overlaid background, mek, edges
#  - picks a number of polar rays and segments a frame using the CM of the background
#  - from segments, it computes: local mek to total mek in frame ratio, local mek to local
#    background ratio, the boundary distance of each segment, and the velocity of each
#    segment; all of these quantities are measured as a function of time (# frame) and
#    segments (segment id)
#  - above data is pickled

# load data: ensure extract_back_mek_edges.py has been run so the correspondence is accurate
type = 'No FGF/MEK1DD'
year = '2019'
IDs = ld.load(type=type, year=year)
root = '/windir/c/Users/Pedrom Zadeh/Documents/Research/Data Storage/'
root = root + '{}Data/{}/XY_Pos'.format(year,type)
ex_id = int(sys.argv[1])
id = IDs.tolist().index(ex_id)
print('Reading tif example ',ex_id)

# holds respective features, all shifted and overlaid to have same CM
# dtype = <list>; dim = [# examples, # frames, 2D image]
print('Reading background...')
with open('JSON Data/background_FGFMEK.json', 'r') as f:
   backs = json.load(f)['backs']
   backs_id = backs[id]
   del(backs)
print('Reading edges...')
with open('JSON Data/edges_FGFMEK.json', 'r') as f:
   edges = json.load(f)['edges']
   edges_id = edges[id]
   del(edges)
print('Reading meks...')
with open('JSON Data/meks_FGFMEK.json', 'r') as f:
   meks = json.load(f)['meks']
   meks_id = meks[id]
   del(meks)

# Get things started
n_segs = 100
n_frames = len(backs_id)
n_dim = (n_frames,n_segs)
vn_dim = (n_frames-1,n_segs)

# mek in this segment / total mek in frame
# mek in this segment / back in this segment
# boundary distances
mekseg_to_mektot, mekseg_to_backseg = np.zeros(shape=n_dim), np.zeros(shape=n_dim)
BDs = np.zeros(shape=n_dim)

# compute ratios + BD
for n in range(1,n_frames):
   print('Computing ratios of activity in frame ', n)
   this_back = np.array(backs_id[n])
   this_mek = np.array(meks_id[n])
   this_edge = np.array(edges_id[n])

   # obtain segment rays from back --> segment back + segment mek
   # do_plot = False
   # if n == 0:
   #    fig, ax = plt.subplots(1,1)
   #    do_plot = True

   lines, cm = obt.polar_rays(this_back,n_segs)
   this_back_segs = obt.seg_via_angle(this_back,lines,cm)
   this_mek_segs = obt.seg_via_angle(this_mek,lines,cm)
   this_edge_segs = obt.seg_via_angle(this_edge,lines,cm)

   for i in range(n_segs):
      plt.imshow(this_back_segs[i])
      plt.show()

   sys.exit(1)

   # if n == 0:
   #    plt.savefig('Visuals/Segmentation Analysis/pos{}/segID.png'.format(str(ex_id)),dpi=400)
   #    plt.close()

   # obtain ratios + BDs
   mekseg_to_mektot[n] = ipt.get_mek_to_mek_ratio(this_mek_segs,this_mek)
   mekseg_to_backseg[n] = ipt.get_mek_to_back_ratio(this_mek_segs,this_back_segs)
   BDs[n] = ipt.get_BD(this_edge_segs,cm)

# compute velocity as d(BD)/dt
velocity = [BDs[t,:] - BDs[t-1,:] for t in range(1,n_frames)]
velocity = np.array(velocity)

# pickle data
mekseg_to_mektot = mekseg_to_mektot.tolist()
mekseg_to_backseg = mekseg_to_backseg.tolist()
BDs = BDs.tolist()
velocity = velocity.tolist()

# make the appropriate directory if it doesn't exist
save_dir = 'JSON Data/pos{}/'.format(ex_id)
if not os.path.exists(save_dir):
   os.mkdir(save_dir)

with open(save_dir + 'ratios_ID{}.json'.format(ex_id,ex_id), 'w') as f:
   json.dump({'localmek_mek' : mekseg_to_mektot, 'mek_back' : mekseg_to_backseg}, f)

with open(save_dir + 'BD_ID{}.json'.format(ex_id,ex_id), 'w') as f:
   json.dump({'BDs' : BDs}, f)

with open(save_dir + 'velocity_ID{}.json'.format(ex_id,ex_id), 'w') as f:
   json.dump({'velocity' : velocity}, f)
