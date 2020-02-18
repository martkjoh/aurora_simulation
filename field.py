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
N = 20
# (x[1,x,y,z],x[2,x,y,z],x[3,x,y,z]) er en vektor plasert i (x, y, z)
x = np.array((np.mgrid[-L:L:N*1j, -L:L:N*1j, -L:L:N*1j]))
# Direction of the dipole
mx, my, mz = 0, 0, 1
dims = (N, N, N)
m = np.array((
    np.ones(dims) * mx,
    np.ones(dims) * my, 
    np.ones(dims) * mz))

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
    return np.array([np.gradient(f, axis = i+1) for i in range(3)]) * N / (2 / L)

def curl(f):
    Df = D(f)
    return np.einsum("ijk,jkxyz->ixyz", eijk, Df)

# Mathematical functions
def A(x):
    r = np.sqrt(dot(x, x))
    return 1/(4*pi) * cross(x/r, m) / r**2

def B(x):
    r = np.sqrt(dot(x, x))
    return 1/(4*pi) * (3*dot(m, x)*x / r**2 - m) / r**3 


# Array manip
# Mask everything not inside radius (r1, r2)
def mask_radialy_vec(f, r1, r2):
    rr = dot(x, x)
    f = np.ma.array(f)
    f = np.ma.array([np.ma.masked_where(rr < r1, f[i]) for i in range(3)])
    f = np.ma.array([np.ma.masked_where(rr > r2, f[i]) for i in range(3)])
    return f

fig = plt.figure()
ax = Axes3D(fig)
ax.grid(False)

Ax = A(x)
Bx = curl(Ax)
B0x = B(x)
print((Bx - B0x) / B0x)

r1 = 1
r2 = 2
Bx = mask_radialy_vec(Bx, r1, r2)    
ax.quiver(*x, *(Bx - B0x), color = "red", pivot = "middle")
plt.show()


# ax.scatter(*x, r2, c = np.ndarray.flatten(rmask))