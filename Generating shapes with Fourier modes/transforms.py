import numpy as np

# consider the first 100 fourier modes
n = 101                                   # total number of fourier modes to consider
ns = np.arange(0,n,1)                     # array of fourier modes

def density_modes(rho_0,tau,T):
   """Returns the fourier modes for mek density, originally modeled as a square patch at origin.
   Ensure to add random angle later to move it to some random place.
   ``rho_0`` -- height of square wave
   ``tau`` -- total width of the square wave
   ``T``  -- period of wave (2pi)"""
   modes = np.zeros(n)
   modes[1:] = 2 * rho_0 * np.sin(np.pi*ns*tau/T)[1:] / (np.pi*ns[1:])
   modes[0] = tau * rho_0 / T
   return modes

def kernel_modes(K_var):
   """Returns the fourier modes for the exponential kernel. ``K_var`` is the variance of the
   kernel in real space."""
   modes = np.exp(-(ns*K_var)**2/2)
   return modes
