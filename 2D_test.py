import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = 1
Y = 1
N = 100
x, y = np.mgrid[-X:X:N*1j, -Y:Y:N*1j]
r = np.array([x, y])

v_x, v_y = 0, 1
v = np.array((np.ones_like(y) * v_x, np.ones_like(x) * v_y))

def dot(r1, r2):
    return np.einsum("ikj,ikj->kj", r1, r2)

def V(r):
    return dot(r, v) / dot(r, r)

def E(r):
    return np.gradient(V(r))

fig, ax = plt.subplots()

ax.streamplot(*r[::-1], *E(r)[::-1])
plt.show()