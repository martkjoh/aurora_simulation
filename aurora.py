import numpy as np
from numpy import pi
from numpy.linalg import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
#
# The numeric constants defining the problem

# magnetic permiablity
mu_0 = 1
# magnetization
m = 1
# Size of the simulated area
X, Y, Z = 1, 1, 1
# Number of points in each dimension
N = 20
# (r[1,x,y,z],r[2,x,y,z],r[3,x,y,z]) er en vektor plasert i (x, y, z)
x, y, z = np.mgrid[-X:X:N*1j, -Y:Y:N*1j, -Z:Z:N*1j]
r = np.array((x, y, z))
# Direction of the dipole
vx, vy, vz = 0, 0, 1
v_dipole = np.array((
    np.ones_like(x) * vx,
    np.ones_like(y) * vy, 
    np.ones_like(z) * vz))

# levi-cevita
eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[2, 0, 1] = eijk[1, 2, 0] = 1
eijk[2, 1, 0] = eijk[0, 2, 1] = eijk[1, 0, 2] = -1


print(eijk.shape, r.shape, v_dipole.shape)
def cross(r1, r2):
    return np.einsum("ijk,jxyz,kxyz->ixyz", eijk, r1, r2)

def dot(r1, r2):
    return np.einsum("ixyz,ixyz->xyz", r1, r2) 

def A(r):
    return mu_0*m/(4*pi) * cross(r, v_dipole) / dot(r, r)**(3 / 2)


fig = plt.figure()
ax = Axes3D(fig)

ax.quiver(*r, *A(r), length = 0.1)
plt.show()