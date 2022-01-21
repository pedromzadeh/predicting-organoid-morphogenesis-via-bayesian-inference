import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import correlation as corr
import transforms as FT
import sys

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

# hyper parameters
N = 50                                     # number of organoids
R0 = 5                                      # initialize each shape to a circle
n = 101                                     # total number of fourier modes to consider
ns = np.arange(0,n,1)                       # array of fourier modes
sigmas = np.zeros(n)                        # noise associated with the nth fourier mode
thetas = np.linspace(-np.pi,np.pi,200)      # thetas

# initialize F[r(theta')], keeping in mind ns=0 needs special treatment
rho0s = [0.1]
taus = [np.pi/4]
Ts = [2*np.pi]
rho_modes_n = [FT.density_modes(rho_0,tau,T) for rho_0,tau,T in zip(rho0s,taus,Ts)]

# initialize F[K]
K_vars = [np.pi/10]
K_modes_n = [FT.kernel_modes(K_var) for K_var in K_vars]

# initialize GFP patch
rho_0 = 0.1
tau = np.pi/4
T = 2*np.pi
GFP_modes = FT.density_modes(rho_0,tau,T)

# initialize modal noise
sigmas[2] = 0.8
# sigmas[2:10] = [0.6, 0.6, 0, 0.8, 0, 0, 0, 0]

# set up parameters for cross correlation computation across N organoids
RR_xc_N, rhorho_xc_N, Rrho_xc_N = [], [], []
RGFP_xc_N, GFPGFP_xc_N = [], []
limit = len(thetas) / 2
shifts_int = np.arange(-limit,limit+1,1).astype('int')     # [-pi:pi] in radians

# begin computation per organoid
for k in range(N):
   print(k)

   # rotate each patch of mek and GFP to a random angle
   M = len(rho_modes_n)
   for m in range(M):
      theta_rot = 2*np.pi*np.random.rand() - np.pi
      rho_modes_n[m] = rho_modes_n[m] * np.exp(1j*ns*theta_rot)

   GFP_theta_rot = 2*np.pi*np.random.rand() - np.pi
   GFP_modes = GFP_modes * np.exp(1j*ns*GFP_theta_rot)

   # add contributions of all mek patches + noise
   cns = [K_modes_n[m] * rho_modes_n[m] for m in range(M)]
   cns = np.array(cns)
   cns = np.sum(cns,axis=0)
   cns += sigmas*np.random.randn(n) + sigmas*np.random.randn(n)*1j
   GFP_qns = GFP_modes

   # compute R(theta) in real space
   # compute rho(theta) in real space
   R_theta = np.ones(len(thetas)) * R0
   rho_theta_n = [np.zeros(len(thetas)) for m in range(M)]
   rho_theta = np.zeros(len(thetas))
   GFP_theta = np.zeros(len(thetas))

   for q in ns:
      R_theta += [(cns[q]*np.exp(1j*q*theta)).real for theta in thetas]
      for m in range(M):
         rho_theta_n[m] += [(rho_modes_n[m][q]*np.exp(1j*q*theta)).real for theta in thetas]
      GFP_theta += [(GFP_qns[q]*np.exp(1j*q*theta)).real for theta in thetas]

   rho_theta_n = np.array(rho_theta_n)
   rho_theta = np.sum(rho_theta_n,axis=0)

   # compute the xc
   RR_xcs = corr.xc(R_theta,R_theta,shifts_int)
   rhorho_xcs = corr.xc(rho_theta,rho_theta,shifts_int)
   Rrho_xcs = corr.xc(R_theta,rho_theta,shifts_int)
   RGFP_xcs = corr.xc(R_theta,GFP_theta,shifts_int)
   GFPGFP_xcs = corr.xc(GFP_theta,GFP_theta,shifts_int)

   # store for later use
   RR_xc_N.append(RR_xcs)
   rhorho_xc_N.append(rhorho_xcs)
   Rrho_xc_N.append(Rrho_xcs)
   RGFP_xc_N.append(RGFP_xcs)
   GFPGFP_xc_N.append(GFPGFP_xcs)

   # plot the shape + overlay rho
   x = R_theta*np.cos(thetas)
   y = R_theta*np.sin(thetas)
   colors = plt.cm.cividis(rho_theta/np.max(rho_theta))
   GFP_colors = plt.cm.bwr(GFP_theta/np.max(GFP_theta))

   for color in GFP_colors:
      if color[0] != 1:
         color[3] = 0
      else:
         color[3] = 0.7

   if k < 25:
      start = k%2
      plt.subplot(5,10,2*k+1)
      plt.scatter(x,y,color=colors,s=0.5)
      plt.scatter(x,y,color=GFP_colors,s=0.5)
      plt.axis('equal')
      plt.axis('off')
      plt.subplot(5,10,2*k+2)
      plt.plot(thetas,R_theta,color='black')

plt.suptitle('Few sample organoids')
# plt.savefig('organoids.png',dpi=800)
plt.show()

# plot correlations averaged over N organoids
xticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
labels= [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$']
RR_xc_N = np.array(RR_xc_N)
rhorho_xc_N = np.array(rhorho_xc_N)
Rrho_xc_N = np.array(Rrho_xc_N)
RGFP_xc_N = np.array(RGFP_xc_N)
GFPGFP_xc_N = np.array(GFPGFP_xc_N)

# average over all organoids
RR_xcs = np.mean(RR_xc_N,axis=0)
rhorho_xcs = np.mean(rhorho_xc_N,axis=0)
Rrho_xcs = np.mean(Rrho_xc_N,axis=0)
RGFP_xcs = np.mean(RGFP_xc_N,axis=0)
GFPGFP_xcs = np.mean(GFPGFP_xc_N,axis=0)

# obtain 25th and 75th percentiles
N = len(shifts_int)
first_quant_GFP = [np.quantile(RGFP_xc_N[:,del_seg],0.25,interpolation='midpoint') for del_seg in range(N)]
second_quant_GFP = [np.quantile(RGFP_xc_N[:,del_seg],0.75,interpolation='midpoint') for del_seg in range(N)]
first_quant_rho = [np.quantile(Rrho_xc_N[:,del_seg],0.25,interpolation='midpoint') for del_seg in range(N)]
second_quant_rho = [np.quantile(Rrho_xc_N[:,del_seg],0.75,interpolation='midpoint') for del_seg in range(N)]

# standard deviations
# first_quant_rho = 2*np.sqrt(np.var(Rrho_xc_N,axis=0))+Rrho_xcs
# second_quant_rho = -2*np.sqrt(np.var(Rrho_xc_N,axis=0))+Rrho_xcs
# first_quant_GFP = 2*np.sqrt(np.var(RGFP_xc_N,axis=0))+RGFP_xcs
# second_quant_GFP = -2*np.sqrt(np.var(RGFP_xc_N,axis=0))+RGFP_xcs

# make plots
x = shifts_int*np.pi/limit
data = [RR_xcs,rhorho_xcs,Rrho_xcs]
ylabels = ['Boundary-Boundary correlation', r'$\rho - \rho$ correlation', r'$\rho$-Boundary correlation']
fig = plt.figure(figsize=(10,5))
for s in range(3):
   plt.subplot(1,3,s+1)
   plt.plot(x,data[s])
   plt.ylabel(ylabels[s])
   plt.xlabel('Angular shift')
   plt.xticks(xticks,labels)
   if s == 2:
      plt.plot(x,first_quant_rho)
      plt.plot(x,second_quant_rho)

plt.tight_layout()
plt.show()

data = [RR_xcs,GFPGFP_xcs,RGFP_xcs]
ylabels = ['Boundary-Boundary correlation', 'GFP-GFP correlation', 'GFP-Boundary correlation']
fig = plt.figure(figsize=(10,5))
for s in range(3):
   plt.subplot(1,3,s+1)
   plt.plot(x,data[s])
   plt.ylabel(ylabels[s])
   plt.xlabel('Angular shift')
   plt.xticks(xticks,labels)
   if s == 2:
      plt.plot(x,first_quant_GFP)
      plt.plot(x,second_quant_GFP)

plt.tight_layout()
plt.show()
