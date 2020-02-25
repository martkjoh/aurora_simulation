import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl

# This file is for setting global parametres used by all scripts

font = {'family' : 'serif', 
        'weight' : 'normal', 
        'size'   : 25}
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rc("lines", lw=2)
plt.rc('font', **font)

# Stepsize for the runge kutta method
dt = 0.005
T = 25
N = int(T / dt)
L = 2

# The magnetic dipole moment of the earth
m = np.array([0.3, -0.1, 1])
m = m / np.sqrt(m @ m)

# Number of particels to start
n_y = 5
n_z = 5

# Initial conditions
XDot0 = np.array([0.2, 0, 0])
X0s = np.empty((n_y * n_z, 3))
for i in range(n_y):
    for j in range(n_z):
        X0s[i * n_y + j, 0] = -2.
        X0s[i * n_y + j, 1] = -1.5 + 0.5*i
        X0s[i * n_y + j, 2]  = 0.1 + 0.3*j
