from parametres import *
from particles import read_path
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos

points = 5
dims = [points] * 3

m = np.array((
    np.ones(dims) * m[0],
    np.ones(dims) * m[1], 
    np.ones(dims) * m[2]))

x = np.array((np.mgrid[
    -L:L:dims[0]*1j, 
    -L:L:dims[1]*1j, 
    -L:L:dims[2]*1j
    ]))

def dot(f1, f2):
    return np.einsum("ixyz,ixyz->xyz", f1, f2)
    
# Returns a 4d array where B_l(x_i, y_j, z_k) = B[l, i, j, k]
def B(x):
    r = np.sqrt(dot(x, x))
    xm = dot(x, m)
    return 1/(4*pi) * (3*xm*x / r**2 - m) / r**3

def plot_earth(ax):
    n = 20
    theta = np.linspace(0, 2*pi, n)
    phi = np.linspace(0, pi, n)
    x = np.array([
        np.outer(cos(theta), sin(phi)),
        np.outer(sin(theta), sin(phi)),
        np.outer(np.ones(n), cos(phi))
    ])
    ax.plot_surface(*x, alpha = 0.1, zorder = 2)
    origin = np.mgrid[0:1:1, 0:1:1, 0:1:1]
    ax.quiver(*origin, *m, pivot = "middle", color = "red", zorder = 1)



def plot_lines3D(ax):
    n = 3 * n_z
    for i in range(n):
        ys = read_path(i)
        ax.plot(*ys[:, 0].T, color = cm.viridis(i/n))

    

# Mask everything not inside radius (r1, r2)
def mask3D(f, r1, r2):
    r = np.sqrt(dot(x, x))
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
    ax.axis("off")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)
    plot_earth(ax)
    plot_lines3D(ax)
    # plot_Bfield(ax, (1, 4))
    ax.pbaspect = [1.0, 1.0, 1.0]

    plt.show()

plot3D()