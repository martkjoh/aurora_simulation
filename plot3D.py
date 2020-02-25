from parametres import *
from particles import read_path, B, dot
from mpl_toolkits.mplot3d import Axes3D
from numpy import sin, cos
from matplotlib.animation import FuncAnimation as FA

points = 8
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


def plot_earth(ax):
    n = 20
    theta = np.linspace(0, 2*pi, n)
    phi = np.linspace(0, pi, n)
    x = np.array([
        np.outer(cos(theta), sin(phi)),
        np.outer(sin(theta), sin(phi)),
        np.outer(np.ones(n), cos(phi))
    ])
    ax.plot_wireframe(*x, alpha = 0.1, color = "green")
    origin = np.mgrid[0:1:1, 0:1:1, 0:1:1]
    ax.quiver(*origin, *m, pivot = "middle", color = "red")

def plot_lines3D(ax):
    n = n_y * n_z
    for i in range(n):
        ys = read_path(i)
        ax.plot(*ys[::10, 0].T, color = cm.viridis(i/n))

# Mask everything not inside radius (r1, r2)
def mask3D(f, r1, r2):
    r = np.sqrt(dot(x, x))
    f = np.ma.array(f)
    f = np.ma.array([np.ma.masked_where(r < r1, f[i]) for i in range(3)])
    f = np.ma.array([np.ma.masked_where(r > r2, f[i]) for i in range(3)])
    return f

# Plot a B field between radius r[0] and r[1]
def plot_Bfield(ax, r):
    Bx = B(x, m)
    Bx = mask3D(Bx, *r)    
    ax.quiver(*x, *Bx, pivot = "middle", length = 0.5, alpha = 0.5)


def plot3D():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.axis("off")
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L / 2, L / 2)
    plot_earth(ax)
    plot_lines3D(ax)
    # cool, but messy
    # plot_Bfield(ax, (1, 4))
    ax.pbaspect = [1.0, 1.0, 1.0]

    plt.show()

def animate():
    fig = plt.figure(figsize=(15, 10))
    ax = Axes3D(fig)
    ax.grid(False)
    ax.set_axis_off()
    l = []
    yss = []
    n = n_y*n_z
    ax.view_init(20, 110)
    for i in range(n):
        yss.append(read_path(i)[1500:])
        plot_earth(ax)
        l.append(ax.plot(*yss[i][:10:10,0].T, color = cm.viridis(i/n))[0])
    
    def anim(i, *fargs):
        n = i * 10
        yss, = fargs
        for j in range(n_y*n_z):        
            x, y, z = (yss[j][:n:10, 0]).T
            l[j].set_data(x, y)
            l[j].set_3d_properties(z)
        return l
    
    a = FA(fig, anim, fargs = (yss, ), frames = N, interval = 10, blit = True)
    plt.show()

# Bonus, if you want it. Works best in interactive mode
# simulate_paths() in main.py must be run first

plot3D()
# animate()