import matplotlib.pyplot as plt
import matplotlib as mpl
import load_module as ld
import numpy as np
import random
import json
import sys

# Read me:
#  - computes centered cross correlations for boundary and ratio arrays for all segments, at a fixed time
#    for all organoids [you can play with what you want BD and R arrays to be really]
#  - plots average and individual centered cross correlations
#  - we don't have enough organoids in 2019. use bootstrap method to augment dataset and estimate variance
#    of such curves [bootstrap is not so great here either, since it only has 5 organoids to sample from]

# matplotlib fonts
params = {'font.family' : 'serif', 'font.size' : 16, 'font.weight' : 'normal', 'image.cmap' : 'cividis'}
mpl.rcParams.update(params)
fig = plt.figure(figsize=(10,7))

# load data
type = 'No FGF/MEK1DD'
year = '2019'
n_segs = 100
n = int(n_segs/2)
time_id = 7
IDs = ld.load(type=type, year=year).tolist()
deltas = np.arange(-n,n+1,1)
xcs = np.zeros(shape=(len(IDs),len(deltas)))
save_dir = 'Visuals/CC/' + type + '/'

# compute for each organoid example, the correlation between boundaries of segments and
# the ratio in each segment. The function is in terms of rotating ratio in segments
# positive deltas indicate CW rotation of ratio array, negative deltas indicate CCW
# rotation of ratio array
# centered covariance isn't being normalized because I don't know if normalizing via
# stds makes sense in this case
for id in IDs:
   if id == 4:
      continue

   print('Computing correlation function for example ', id)
   load_dir = 'JSON Data/pos{}/'.format(id)
   with open(load_dir + 'ratios_ID{}.json'.format(id), 'r') as f:
      ratio_dict = json.load(f)
   localmek_mek, mek_back  = ratio_dict['localmek_mek'], ratio_dict['mek_back']

   with open(load_dir + 'BD_ID{}.json'.format(id), 'r') as f:
      BDs = json.load(f)['BDs']

   BD = np.array(BDs)[time_id,:]
   ratio = np.array(mek_back)[time_id,:]
   # BD = ratio
   # ratio = np.random.uniform(0,1,len(BD))
   # ratio = BD

   covs = []
   x_mean = np.mean(BD)
   x_std = np.sqrt(np.var(BD))
   for delta in deltas:
      Y = np.roll(ratio,delta)
      y_mean = np.mean(Y)
      y_std = np.sqrt(np.var(Y))

      cov = np.mean((BD - x_mean) * (Y - y_mean))
      covs.append(cov)

   xcs[IDs.index(id)] = covs

# type of XC plot being made
xc_type = 'BDmek'
# mean from true sample
mean_true = np.mean(xcs,axis=0)
plt.plot(deltas,mean_true,lw=3,color='black',label='Mean from original sample')
[plt.plot(deltas,xcs[IDs.index(i),:],alpha=0.2,label='organoid {}'.format(i),color='black') for i in IDs]
plt.legend()
plt.title('Centered cross correlation computed from {} organoids at time {} hours'.format(len(IDs),time_id*12))
plt.ylabel(r'$\sum_k (X[k]-\overline{X})(Y[k+\delta_{segment}]-\overline{Y})$')
plt.xlabel(r'$\delta_{segment}$')
# plt.savefig(save_dir + 'XC_{}.png'.format(xc_type), dpi=800)
plt.close()
# plt.show()


# bootstrap to estimate variance
N = 1000
n_deltas = len(deltas)
n_sample = len(IDs)
agg_data = []

fig = plt.figure(figsize=(20,10))
for n in range(N):
   resample_w_replacement = np.array(random.choices(xcs,k=n_sample)).reshape(n_sample,n_deltas)
   mean_n = np.mean(resample_w_replacement,axis=0)
   if n == 0:
      plt.plot(deltas,mean_n,alpha=0.2,lw=1,color='black',label='Mean from resample w/ replacement')

   plt.plot(deltas,mean_n,alpha=0.2,lw=1,color='black')
   agg_data.append(mean_n)

plt.plot(deltas,mean_true,lw=3,color='red',label='Mean from original sample')
plt.title('Estimating variance of mean correlation across organoids via MC bootstrapping')
plt.ylabel(r'$\langle \mathrm{cross\ correlation} \rangle_{\mathrm{organoid}}$')
plt.xlabel(r'$\delta_{segment}$')
plt.legend()
# plt.savefig(save_dir + 'bootstrap_{}.png'.format(xc_type), dpi=800)
plt.close()
# plt.show()

# plot 25th, 50th, 75th quantiles + average from original sample
agg_data = np.array(agg_data)
first_quant = [np.quantile(agg_data[:,del_seg],0.25,interpolation='midpoint') for del_seg in range(n_deltas)]
second_quant = [np.quantile(agg_data[:,del_seg],0.5,interpolation='midpoint') for del_seg in range(n_deltas)]
third_quant = [np.quantile(agg_data[:,del_seg],0.75,interpolation='midpoint') for del_seg in range(n_deltas)]
fig = plt.figure(figsize=(14,7))
plt.title('Variance obtained from MC bootstrap')
plt.ylabel(r'$\langle \mathrm{cross\ correlation} \rangle_{\mathrm{organoid}}$')
plt.xlabel(r'$\delta_{segment}$')
plt.plot(deltas,first_quant,label='25th quantile',color='red',alpha=0.3,lw=2)
plt.plot(deltas,second_quant,label='50th quantile',color='blue',alpha=0.3,lw=2)
plt.plot(deltas,third_quant,label='75th quantile',color='green',alpha=0.3,lw=2)
plt.plot(deltas,mean_true,label='Mean from otiginal sample',color='black',lw=2)
plt.legend()
plt.savefig(save_dir + 'variance_{}.png'.format(xc_type), dpi=800)
plt.close()
# plt.show()
