from particles import *
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

points = 10
dims = [11, 1, 11]

m = np.array((
    np.ones(dims) * m[0],
    np.ones(dims) * m[1], 
    np.ones(dims) * m[2]))

x = np.array((np.mgrid[
    -L:L:dims[0]*1j, 
    -0:0:dims[1]*1j, 
    -L:L:dims[2]*1j
    ]))

def dot(f1, f2):
    return np.einsum("ixyz,ixyz->xyz", f1, f2)
    
# Returns a 4d array where B_l(x_i, y_j, z_k) = B[l, i, j, k]
def B(x):
    r = np.sqrt(dot(x, x))

    xm = dot(x, m)
    return 1/(4*pi) * (3*xm*x / r**2 - m) / r**3


def plot_lines3D(ax, lines):
    n = len(lines)
    for i in range(n):
        ax.pbaspect = [1.0, 1.0, 1.0]
        ax.plot(*lines[i][:,0].T, color = cm.viridis(i/n))

    origin = np.mgrid[0:1:1, 0:1:1, 0:1:1]
    plt.quiver(*origin, *m, pivot = "middle", color = "red")
    # ax.scatter(*origin, s = 5000)

# Mask everything not inside radius (r1, r2)
def mask3D(f, r1, r2):
    r = np.sqrt(dot(x, x))
    print(f.shape)
    f = np.ma.array(f)
    f = np.ma.array([np.ma.masked_where(r < r1, f[i]) for i in range(3)])
    f = np.ma.array([np.ma.masked_where(r > r2, f[i]) for i in range(3)])
    return f

# Plot a B field between radius r[0] and r[1]
def plot_Bfield(ax, r):
    Bx = B(x)
    Bx = mask3D(Bx, *r)    
    ax.quiver(*x, *Bx, pivot = "middle", length = 2)


def plot3D():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.pbaspect = [1.0, 1.0, 1.0]
    ax.axis("off")

    # lines = send_particles()
    # plot_lines3D(ax, lines)
    plot_Bfield(ax, (0.6, 2))

    plt.show()

plot3D()