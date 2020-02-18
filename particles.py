import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

dt = 0.005
T = 15
N = int(T / dt)
L = 2

# The magnetic dipole moment
m = np.array([-0.02, 0.02, 1.2])
m = m / np.sqrt(m @ m)

# Number of points in the discrete vec.field
points = 10
dims = [points] * 3
m_field = np.array((
    np.ones(dims) * m[0],
    np.ones(dims) * m[1], 
    np.ones(dims) * m[2]))

x_field = np.array((np.mgrid[-L:L:points*1j, -L:L:points*1j, -L:L:points*1j]))

def RK4(ys, f, i):
    k1 = f(ys[i]) * dt
    k2 = f(ys[i] + k1 / 2) * dt
    k3 = f(ys[i] + k2 / 2) * dt
    k4 = f(ys[i] + k3) * dt
    delta = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    ys[i + 1]  = ys[i] + delta

# Gives the B-field at a point
def B(x):
    r = np.sqrt(x @ x)
    return (1/ 4 * pi) * (3*(m @ x) * x / r**2 - m) / r**3


def dot_fields(f1, f2):
    return np.einsum("ixyz,ixyz->xyz", f1, f2)

# Returns a 4d array where B_l(x_i, y_j, z_k) = B[l, i, j, k]
def B_field(x):
    r = np.sqrt(dot_fields(x, x))
    xm = dot_fields(x, m_field)
    return 1/(4*pi) * (3*xm*x / r**2 - m_field) / r**3

def f(y):
    return np.array([y[1], np.cross(y[1], B(y[0]))])

# Array manip
# Mask everything not inside radius (r1, r2)
def mask_field(f, r1, r2):
    r = np.sqrt(dot_fields(x_field, x_field))
    f = np.ma.array(f)
    f = np.ma.array([np.ma.masked_where(r < r1, f[i]) for i in range(3)])
    f = np.ma.array([np.ma.masked_where(r > r2, f[i]) for i in range(3)])
    return f

def send_particles():
    a = 4
    b = 5
    lines = []
    for i in range(a):
        for j in range(b):
            # y[i, j, k] is the k'th coordinat of the j'th derivative of x wrt t at t = dt*i
            ys = np.empty((N + 1, 2, 3))
            # starting condititons
            X0 = [-2, -1 + 0.5*j, 0.1 + 0.3*i]
            XDot0 = [0.2, 0, 0]
            y0 = np.array([X0, XDot0])

            ys[0] = y0

            for k in range(N):
                RK4(ys, f, k)
            
            lines.append(ys)
    return lines

def plot_lines3D(ax, lines):
    n = len(lines)
    for i in range(n):
        ax.pbaspect = [1.0, 1.0, 1.0]
        ax.plot(*lines[i][:,0].T, color = cm.viridis(i/n))

    origin = np.mgrid[0:1:1, 0:1:1, 0:1:1]
    plt.quiver(*origin, *m, pivot = "middle", color = "red")
    ax.scatter(*origin, s = 5000)

# Plot a B field between radius r[0] and r[1]
def plot_Bfield(ax, r):
    Bx = B_field(x_field)
    Bx = mask_field(Bx, *r)    
    ax.quiver(*x_field, *Bx, pivot = "middle", length = 2)


fig = plt.figure()
ax = Axes3D(fig)
ax.pbaspect = [1.0, 1.0, 1.0]
ax.axis("off")

lines = send_particles()
plot_lines3D(ax, lines)
plot_Bfield(ax, (0.6, 2))

plt.show()
