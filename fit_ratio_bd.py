import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
import json
import sys

# Read me:
#  - perform regressions on mek/back vs BD data for select segments as function of time
#  - global regression: linear + using all BD and all mek/back data at once
#  - local regression: linear or m-neighboring regression
#  - makes plots of the result and saves to Visuals/
#
# Note: m = 1 suffers in STAN due to the early on zeros in ratio array, so alpha and beta
#       cannot be fitted for as accuretly --> leads to diverging local predictions

# return predicted boundary distances with input from Stan regression
def predicted_BD(pars, segs, n_frames,n_segs):
   """
      pars --- dict of fit parameters
      segs --- list of selected segments to analyze
      n_frames --- number of frames (aka time)
      n_segs --- number of total segments in a frame

      Returns: ndarray of predicted BD for all selected segments, size is same as BD,
               zero everywhere other than selected segments for analysis
   """

   A, B, alpha, beta = pars['A'], pars['B'], pars['alpha'], pars['beta']
   bds = np.zeros(shape=(n_frames,n_segs))
   for seg_id in segs:
      pos_ratio, neg_ratio = np.zeros(shape=(n_frames)), np.zeros(shape=(n_frames))

      for q in range(m):
         pos_ratio += mek_back[:,seg_id+1+q]
         neg_ratio += mek_back[:,seg_id-1-q]

      bds[:,seg_id] = B + A * mek_back[:,seg_id] + alpha * pos_ratio + beta * neg_ratio

   return bds

# load STAN models
try:
   with open('models.pkl', 'rb') as f:
      model_dict = pickle.load(f)
   linear_stan_model = model_dict['linear_model']
   mn_stan_model = model_dict['mn_model']
except:
   raise ValueError('No STAN models found to load. Make sure models.pkl exists and is up-to-date.')

# matplotlib fonts
params = {'font.family' : 'serif', 'font.size' : 12, 'font.weight' : 'normal', 'image.cmap' : 'cividis'}
mpl.rcParams.update(params)

# load data
ex_id = str(sys.argv[1])
with open('JSON Data/pos{}/ratios_ID{}.json'.format(ex_id,ex_id), 'r') as f:
   mek_back = json.load(f)['mek_back']

with open('JSON Data/pos{}/BD_ID{}.json'.format(ex_id,ex_id), 'r') as f:
   BDs = json.load(f)['BDs']

# get a few things ready [m := number of neighbors]
BDs = np.array(BDs)
mek_back = np.array(mek_back)
n_frames = len(BDs)
n_segs = len(BDs[0])
seg_step = 10
times = np.arange(0,n_frames,1)
segs = np.arange(seg_step,n_segs,seg_step)
m = 4

# global linear regression
global_BD = BDs[:,segs].flatten()
global_ratio = mek_back[:,segs].flatten()
data = {'N' : len(global_BD), 'BD' : global_BD, 'R' : global_ratio}
global_fit = linear_stan_model.sampling(data=data, iter=2000, chains=4, warmup=500, thin=1, seed=101)
A = np.mean(global_fit['A'])
B = np.mean(global_fit['B'])
predict_global_BD = global_ratio * A + B

# local regressions
fig, axes = plt.subplots(3,3,figsize=(18,10))
fig.suptitle('Relationship between boundary distance and mek/back for organoid {} (m = {})'.format(ex_id,m), fontweight='bold')
right = 0.85
top = 0.85
wspace = 0.5
hspace = 0.45

data = {'N' : len(segs), 'm' : m, 'n_seg' : n_segs, 'n_time' : n_frames, 'seg_ids' : segs, 'BD' : BDs[:,segs], 'ratio' : mek_back}
fit = mn_stan_model.sampling(data=data, iter=1000, chains=4, warmup=500, thin=1, seed=101)
print(fit)

A = np.mean(fit['A'])
B = np.mean(fit['B'])
alpha = np.mean(fit['alpha'])
beta = np.mean(fit['beta'])
pars = dict(A=A, B=B, alpha=alpha, beta=beta)
bd_pred = predicted_BD(pars,segs,n_frames,n_segs)

for seg_id in segs:

   k = segs.tolist().index(seg_id)
   i = int(k / 3)
   j = k % 3

   if j == 0:
      axes[i,j].set_ylabel('Boundary Distance')
   if i == 2:
      axes[i,j].set_xlabel('Mek/Back')

   axes[i,j].set_title('segment = {}'.format(seg_id))
   axes[i,j].scatter(mek_back[:,seg_id], bd_pred[:,seg_id],alpha=0.3,label='Local prediction',color='orange')
   axes[i,j].plot(global_ratio, predict_global_BD, lw=2, color='green', alpha=0.5,label='Global prediction')
   axes[i,j].scatter(mek_back[:,seg_id], BDs[:,seg_id],alpha=0.3, label='Observed')

   handles, labels = axes[i,j].get_legend_handles_labels()

fig.legend(handles,labels,loc='center right')
plt.subplots_adjust(right=right, top=top, wspace=wspace, hspace=hspace)
plt.savefig('Visuals/Segmentation Analysis/pos{}/ratio_bd_regression_m{}.png'.format(ex_id,m), dpi=600)
# plt.show()
