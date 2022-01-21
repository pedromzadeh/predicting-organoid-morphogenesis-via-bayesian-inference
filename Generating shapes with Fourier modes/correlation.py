import numpy as np
import sys

# Read me:
# Module for computing cross correlations between X and Y arrays via deltas

def xc(X,Y,deltas,norm=False):
   """Compute the cross correlation(xc) between ``X`` and ``Y`` via ``deltas``."""
   xcs = []
   x_mean = np.mean(X)
   x_std = np.sqrt(np.var(X))
   for delta in deltas:
      Y_shifted = np.roll(Y,delta)
      Y_shifted_mean = np.mean(Y_shifted)
      Y_shifted_std = np.sqrt(np.var(Y_shifted))

      if norm:
         xc = np.mean((X - x_mean) * (Y_shifted - Y_shifted_mean)) / (x_std*Y_shifted_std)
      else:
         xc = np.mean((X - x_mean) * (Y_shifted - Y_shifted_mean))

      xcs.append(xc)
   return xcs


# time_id = 4
# # mean from true sample
# mean_true = np.mean(corrs,axis=0)
# plt.plot(deltas,mean_true,lw=3,color='black',label='Mean from original sample')
# [plt.plot(deltas,corrs[IDs.index(i),:],alpha=0.2,color='black',lw=1) for i in IDs]
