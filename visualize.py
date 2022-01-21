import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as tick
import pandas as pd
import numpy as np
import json
import sys
import os

# Read me:
#  - loads the two flavor of ratios, boundary distances, and velocities
#  - makes kymographs as functions of time and segment
#  - for select times, plots BD and ratio as a function of segments
#     + plots correlation functions of these two to see if there's any chirality
#  - for select segments, plots BD and ratio as a function of time
#     + see potential linear relationship --> see fit_ratio_bd.py for regression

# matplotlib fonts
params = {'font.family' : 'serif', 'font.size' : 12, 'font.weight' : 'normal', 'image.cmap' : 'cividis'}
mpl.rcParams.update(params)

# load data
ex_id = str(sys.argv[1])
print('Making plots for organoid ', ex_id)
load_dir = 'JSON Data/pos{}/'.format(ex_id)
save_dir = 'Visuals/Segmentation Analysis/pos{}/'.format(ex_id)
if not os.path.exists(load_dir):
   raise ValueError('Make sure the pos/ directory is available and not empty.')
   sys.exit(1)

# make the appropriate directory if it doesn't exist
if not os.path.exists(save_dir):
   os.mkdir(save_dir)

with open(load_dir + 'ratios_ID{}.json'.format(ex_id,ex_id), 'r') as f:
   ratio_dict = json.load(f)
localmek_mek, mek_back  = ratio_dict['localmek_mek'], ratio_dict['mek_back']

with open(load_dir + 'BD_ID{}.json'.format(ex_id,ex_id), 'r') as f:
   BDs = json.load(f)['BDs']

with open(load_dir + 'velocity_ID{}.json'.format(ex_id,ex_id), 'r') as f:
   velocity = json.load(f)['velocity']

# plot ratio related visuals
plt.xlabel('Segments')
plt.ylabel('Time (frame ID)')
plt.imshow(localmek_mek,origin='lower')
cb = plt.colorbar()
cb.set_label('Ratio of local MEK to total MEK in frame')
plt.axis('tight')
plt.savefig(save_dir + 'localmek_mek.png'.format(ex_id),dpi=400)
plt.close()

plt.xlabel('Segments')
plt.ylabel('Time (frame ID)')
plt.imshow(mek_back,origin='lower')
cb = plt.colorbar()
cb.set_label('Ratio of local MEK to local background')
plt.axis('tight')
plt.savefig(save_dir + 'localmek_back.png'.format(ex_id),dpi=400)
plt.close()

# plot BD
plt.xlabel('Segments')
plt.ylabel('Time (frame ID)')
plt.imshow(BDs,origin='lower')
cb = plt.colorbar()
cb.set_label('Boundary distance w.r.t frame\'s CM')
plt.axis('tight')
plt.savefig(save_dir + 'BD.png'.format(ex_id),dpi=400)
plt.close()

# plot velocity
plt.xlabel('Segments')
plt.ylabel('Time (frame ID)')
plt.imshow(velocity,origin='lower')
cb = plt.colorbar()
cb.set_label(r'Velocity ($v=d(BD)/dt$)')
plt.axis('tight')
plt.savefig(save_dir + 'velocity.png'.format(ex_id),dpi=400)
plt.close()

# plots for select times
n_frames = len(BDs)
time_step = 1
times = np.arange(0,n_frames,time_step)
segs = np.arange(0,len(BDs[0]))
BDs = np.array(BDs)
mek_back = np.array(mek_back)

fig, axes = plt.subplots(2,4,figsize=(18,8))
right = 0.8
top = 0.9
wspace = 0.4
hspace = 0.3
fig.suptitle('Comparison for organoid {}'.format(ex_id),fontweight='bold')

for t in times:
   bd_t = BDs[t,:]
   r_t = mek_back[t,:]

   i = int(times.tolist().index(t) / 4)
   j = times.tolist().index(t) % 4

   axes[i,j].set_title('time = {}'.format(t))
   axes[i,j].scatter(segs,bd_t,alpha=0.3,label='Boundary Distance',color='red')
   ax2 = axes[i,j].twinx()
   ax2.scatter(segs,r_t,alpha=0.3,label='Ratio of mek to back')

   if j == 0:
      axes[i,j].set_ylabel('Boundary Distance')
   if j == len(axes[i])-1:
      ax2.set_ylabel('Mek/Back')
   if i == 1:
      axes[i,j].set_xlabel('Segments')

   pt_1, label_1 = axes[i,j].get_legend_handles_labels()
   pt_2, label_2 = ax2.get_legend_handles_labels()
   pts = (pt_1[0], pt_2[0])
   labels = (label_1[0], label_2[0])

fig.legend(pts,labels,loc='center right')
plt.tight_layout()
plt.subplots_adjust(right=right, top=top, wspace=wspace, hspace=hspace)
plt.savefig(save_dir + 'bd_ratio_segs.png'.format(ex_id),dpi=600)
plt.close()

# plots for select segments [not useful with 8 (2019) or 12 (2020) frames]
# n_frames = len(BDs)
# n_segs = len(BDs[0])
# seg_step = 10
# times = np.arange(0,n_frames,1)
# segs = np.arange(0,n_segs,seg_step)
# fig, axes = plt.subplots(2,5,figsize=(18,8))
# right = 0.8
# top = 0.9
# wspace = 0.5
# hspace = 0.3
# fig.suptitle('Comparison for organoid {}'.format(ex_id),fontweight='bold')
#
# for seg_id in segs:
#    bd_seg = BDs[:,seg_id]
#    r_seg = mek_back[:,seg_id]
#
#    i = int(segs.tolist().index(seg_id) / 5)
#    j = segs.tolist().index(seg_id) % 5
#
#    axes[i,j].set_title('segment = {}'.format(seg_id))
#    axes[i,j].scatter(times,bd_seg,alpha=0.3,label='Boundary Distance',color='red')
#    ax2 = axes[i,j].twinx()
#    ax2.scatter(times,r_seg,alpha=0.3,label='Ratio of mek to back')
#    ax2.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
#
#    if j == 0:
#       axes[i,j].set_ylabel('Boundary Distance')
#    if j == len(axes[i])-1:
#       ax2.set_ylabel('Mek/Back')
#    if i == 1:
#       axes[i,j].set_xlabel('Time')
#
#    pt_1, label_1 = axes[i,j].get_legend_handles_labels()
#    pt_2, label_2 = ax2.get_legend_handles_labels()
#    pts = (pt_1[0], pt_2[0])
#    labels = (label_1[0], label_2[0])
#
# fig.legend(pts,labels,loc='center right')
# plt.tight_layout()
# plt.subplots_adjust(right=right, top=top, wspace=wspace, hspace=hspace)
# plt.savefig(save_dir + 'bd_ratio_time.png'.format(ex_id),dpi=600)
# plt.close()
