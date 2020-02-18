import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

m = np.array([-0.02, 0.02, 1.2])
m = m / np.sqrt(m @ m)
dt = 0.001
T = 2
N = int(T / dt)
L = 2


def RK4(ys, f, i):
    k1 = f(ys[i]) * dt
    k2 = f(ys[i] + k1 / 2) * dt
    k3 = f(ys[i] + k2 / 2) * dt
    k4 = f(ys[i] + k3) * dt
    delta = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ys[i + 1]  = ys[i] + delta


def B(x):
    r = x @ x
    return (1/ 4 * pi) * (3*(m.T @ x) * x / r**2 - m) / r**3

def f(y):
    return np.array([y[1], np.cross(y[1], B(y[0]))])


fig = plt.figure()
ax = Axes3D(fig)
ax.axis("off")
ax.pbaspect = [1.0, 1.0, 1.0]
a = 4
b = 5
for i in range(a):
    for j in range(b):
        # y[i, j, k] is the k'th coordinat of the j'th derivative of x wrt t at t = dt*i
        ys = np.empty((N + 1, 2, 3))

        X0 = [-1, -0.2 + 0.1*j, 0.3 + 0.1*i]
        XDot0 = [1, 0, 0]
        y0 = np.array([X0, XDot0]) * L

        ys[0] = y0

        for k in range(N):
            RK4(ys, f, k)

        ax.plot(*ys[:,0].T, color = cm.viridis((i*b + j)/(a*b)))


A = np.mgrid[0:1:1, 0:1:1, 0:1:1]
plt.quiver(*A, *m, pivot = "middle")
plt.show()

# class Particle:
#     def __init__(self, x, xDot):
#         self.x = x
#         self.xDot = xDot
