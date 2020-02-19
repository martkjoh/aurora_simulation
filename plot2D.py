from particles import *
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

n = 20
L = 2
x = np.array(np.mgrid[-L:L:n*1j, -L:L:n*1j])

def dot(x1, x2):
    return np.einsum("ixy, ixy -> xy", x1, x2)

def B2D(x, i, j):
    mx =  x[0] * m[i] + x[1] * m[j]
    m2D = np.array([m[i]*np.ones_like(x[0]), m[j] * np.ones_like(x[1])])
    r = np.sqrt(dot(x, x))
    return (1/ 4*pi) * (3*mx*x / r**2 - m2D) / r**3

def mask2D(f, R):
    r = np.sqrt(dot(x, x))
    f = np.ma.array(f)
    return np.ma.array([np.ma.masked_where(r < R, f[i]) for i in range(2)])


def plot_field2D(ax):
    Bxz = mask2D(B2D(x, 0, 2), 0.8)
    Byz = mask2D(B2D(x, 1, 2), 0.8)
    ax[0].quiver(*x, *Bxz, pivot = "middle", alpha = 0.2)
    ax[1].quiver(*x, *Byz, pivot = "middle", alpha = 0.2)

def plot2D():
    lines = get_lines()
    for i in range(n_y):
        fig, ax = plt.subplots(1, 2, sharey=True)
        ax[0].set_xlabel("$x$")
        ax[0].set_ylabel("$z$")
        ax[1].set_xlabel("$y$")
        ax[1].set_ylabel("$z$")
        for j in range(n_z):
            k = i*n_z + j
            ys = lines[k]
            ax[0].plot(ys[:, 0, 0], ys[:, 0, 2], color = cm.viridis(j / n_z))
            ax[1].plot(ys[:, 0, 1], ys[:, 0, 2], color = cm.viridis(j / n_z))
            plot_field2D(ax)

        plt.show()

plot2D()