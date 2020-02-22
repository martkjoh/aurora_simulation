import numpy as np
from numpy import pi

# This file is for setting global parametres used by all scripts

# Stepsize for the runge kutta method
dt = 0.01
T = 15
N = int(T / dt)
L = 2

# The magnetic dipole moment of the earth
m = np.array([0.3, -0.1, 1])
m = m / np.sqrt(m @ m)

# Number of particels to start
n_y = 5
n_z = 4

# Initial conditions
XDot0 = [0.2, 0, 0]
X0s = []
for i in range(n_y):
    for j in range(n_z):
        X0s.append([-2, -1 + 0.5*i, 0.1 + 0.3*j])
