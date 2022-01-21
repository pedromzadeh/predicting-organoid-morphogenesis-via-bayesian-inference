import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import correlation as corr
import transforms as FT
import sys
import json
import time
import multiprocessing

# Read me:
# The goal is to generate organoid shapes via fourier modes. We hope that by having
# and analytical formula, we can better control and understand features that show up
# in the correlation plots. The mathematical formulation behind this process is in
# OneNote. Here we compute the entire organoid shape from only ONE patch of
# mek density residing at a fixed (randomly determined) angle. The model is a square wave.
# modeled as a square wave. Also note that the nth fourier mode introduces an enclosed
# shape with n bumps.
# Further note: noise has the physical meaning of "boundary protrusions caused by means
# other than mek density".


def run(N,R0,noise,rho_0):

   print('rho0s = ' , rho_0)

   # hyper parameters
   n = 101
   ns = np.arange(0,n,1)
   sigmas = np.zeros(n)
   thetas = np.linspace(-np.pi,np.pi,200)

   # initialize F[r(theta')], keeping in mind ns=0 needs special treatment
   rho0s = rho_0
   taus = [np.pi/4,np.pi/4]
   Ts = [2*np.pi,2*np.pi]
   rho_modes_n = [FT.density_modes(rho_0,tau,T) for rho_0,tau,T in zip(rho0s,taus,Ts)]

   # initialize F[K]
   K_vars = [np.pi/10,np.pi/10]
   K_modes_n = [FT.kernel_modes(K_var) for K_var in K_vars]

   # initialize 2nd FM noise
   sigmas[2:3] = [noise]

   # set up parameters for cross correlation computation across N organoids
   Rrho_xc_N = []
   limit = len(thetas) / 2
   shifts_int = np.arange(-limit,limit+1,1).astype('int')     # [-pi:pi] in radians

   # begin computation per organoid
   for k in range(N):

      # rotate each patch of mek and GFP to a random angle
      M = len(rho_modes_n)
      for m in range(M):
         theta_rot = 2*np.pi*np.random.rand() - np.pi
         rho_modes_n[m] = rho_modes_n[m] * np.exp(1j*ns*theta_rot)

      # add contributions of all mek patches + noise
      cns = [K_modes_n[m] * rho_modes_n[m] for m in range(M)]
      cns = np.array(cns)
      cns = np.sum(cns,axis=0)
      cns += sigmas*np.random.randn(n) + sigmas*np.random.randn(n)*1j

      R_theta = np.ones(len(thetas)) * R0
      rho_theta_n = [np.zeros(len(thetas)) for m in range(M)]
      rho_theta = np.zeros(len(thetas))

      for q in ns:
         R_theta += [(cns[q]*np.exp(1j*q*theta)).real for theta in thetas]
         for m in range(M):
            rho_theta_n[m] += [(rho_modes_n[m][q]*np.exp(1j*q*theta)).real for theta in thetas]

      rho_theta_n = np.array(rho_theta_n)
      rho_theta = np.sum(rho_theta_n,axis=0)

      # compute the xc
      Rrho_xcs = corr.xc(R_theta,rho_theta,shifts_int)

      # store for later use
      Rrho_xc_N.append(Rrho_xcs)

   # average over all organoids
   Rrho_xc_N = np.array(Rrho_xc_N)
   Rrho_xcs = np.mean(Rrho_xc_N,axis=0)

   # obtain 25th and 75th percentiles
   l = len(shifts_int)
   first_quant_rho = [np.quantile(Rrho_xc_N[:,del_seg],0.25,interpolation='midpoint') for del_seg in range(l)]
   second_quant_rho = [np.quantile(Rrho_xc_N[:,del_seg],0.75,interpolation='midpoint') for del_seg in range(l)]

   mid_index = int(len(Rrho_xcs)/2)
   quant1, mean, quant2 = first_quant_rho[mid_index], Rrho_xcs[mid_index], second_quant_rho[mid_index]

   # statistically significantly above 0
   if quant1 > 10E-5 and mean > 0 and quant2 > 0:
      # x = shifts_int*np.pi/limit
      # fig = plt.figure(figsize=(10,5))
      # plt.plot(x,Rrho_xcs)
      # plt.plot(x,first_quant_rho)
      # plt.plot(x,second_quant_rho)
      #
      # plt.tight_layout()
      # plt.show()

      return 1
   else:
      return 0

# Simulate
t0 = time.time()
N = 500
noises = np.linspace(0,0.9,19)   # 2nd FM noise
radii = np.arange(1,11,1)
rho0s = np.linspace(0.05,1,20)
DN = noises.shape[0]
DR = rho0s.shape[0]
phasediagram = np.zeros(shape=(DR,DR)).astype('int')

print(noises)
print(radii)
print(rho0s)

# fixed noise at 0.3, 0.7
R0 = 2
noise = 0.7
for rho0 in rho0s:
   print('rho0 = ', rho0)
   pool = multiprocessing.Pool(processes=8)

   i = rho0s.tolist().index(rho0)
   args = [(N,R0,noise,[rho0,rho0_2]) for rho0_2 in rho0s]
   list = pool.starmap(run,args)
   phasediagram[i] = list
   print(phasediagram)

with open('phasediagram.json', 'w') as f:
   json.dump({'PD' : phasediagram.tolist()},f)

print(time.time()-t0)
