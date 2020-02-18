import numpy as np
from numpy import pi
from numpy.linalg import norm
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Constants
#
# The numeric constants defining the problem

# Size of the simulated area
L = 1
# Number of points in each dimension
N = 10
# (r[1,x,y,z],r[2,x,y,z],r[3,x,y,z]) er en vektor plasert i (x, y, z)
x, y, z = np.mgrid[-L:L:N*1j, -L:L:N*1j, -L:L:N*1j]
r = np.array((x, y, z))
# Direction of the dipole
mx, my, mz = 0, 0, 1
m = np.array((
    np.ones_like(x) * mx,
    np.ones_like(y) * my, 
    np.ones_like(z) * mz))

# levi-cevita
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[2, 0, 1] = eijk[1, 2, 0] = 1
eijk[2, 1, 0] = eijk[0, 2, 1] = eijk[1, 0, 2] = -1


# Vector operations
def cross(r1, r2):
    return np.einsum("ijk,jxyz,kxyz->ixyz", eijk, r1, r2)

def dot(r1, r2):
    return np.einsum("ixyz,ixyz->xyz", r1, r2) 

def D(f):
    return np.array([np.gradient(f, axis = i+1) for i in range(3)])

def curl(f):
    Df = D(f)
    return np.einsum("ijk,jkxyz->ixyz", eijk, Df)

# Mathematical functions
def A(r):
    return 1/(4*pi) * cross(r, m) / dot(r, r)**(3 / 2)


# Array manip
# Mask everything not inside radius (r1, r2)
def mask_radialy_vec(f, r1, r2):
    rr = dot(r, r)
    f = np.ma.array(f)
    f = np.ma.array([np.ma.masked_where(rr < r1, f[i]) for i in range(3)])
    f = np.ma.array([np.ma.masked_where(rr > r2, f[i]) for i in range(3)])
    return f

fig = plt.figure()
ax = Axes3D(fig)
ax.grid(False)

Ar = A(r)
Br = curl(Ar)

r1 = 0.1
r2 = 1
Ar = mask_radialy_vec(Ar, r1, r2)
Br = mask_radialy_vec(Br, r1, r2)    

ax.quiver(*r, *Ar, length = 0.2, pivot = "middle")
ax.quiver(*r, *Br, length = 0.2, color = "red", pivot = "middle")
# ax.scatter(*r, r2, c = np.ndarray.flatten(rmask))

plt.show()