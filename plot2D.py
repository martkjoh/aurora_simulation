from parametres import *
from particles import read_path


n = 20
L = 2
x = np.array(np.mgrid[-L:L:n*1j, -L:L:n*1j])
legends = ["$\\vec B/[B_0]$", "$\\vec m$",]

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


def plot_field(ax, alpha = 1):
    # Getting the xz and yz componenes of the field and dp-moment
    Bs = [mask2D(B2D(x, 0, 2), 1), mask2D(B2D(x, 1, 2), 1)]
    ms = [[m[0], m[2]], [m[1], m[2]]]
    origin = np.mgrid[0:1:1, 0:1:1]
    legend_element = []
    for i in range(2):
        legend_element.append(ax[i].quiver(*x, *Bs[i], pivot = "middle", alpha = 0.9 *alpha))
        earth = plt.Circle((0, 0), 1, color = "blue", alpha = 0.2 * alpha)
        ax[i].add_artist(earth)
        legend_element.append(ax[i].quiver(*origin, *ms[i], scale = 2, pivot = "middle", color = "red", alpha = 0.8 * alpha))

    return legend_element

# Plot one row of particle-lines (same y-value)
def plot_lines(ax, i):
    for j in range(n_z):
        k = i*n_z + j
        ys = read_path(k)
        ax[0].plot(ys[:, 0, 0], ys[:, 0, 2], color = cm.viridis(j / n_z))
        ax[1].plot(ys[:, 0, 1], ys[:, 0, 2], color = cm.viridis(j / n_z))

def get_ax():
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize = (20, 10))
    ax[0].set_xlabel("$x/[R_0]$")
    ax[0].set_ylabel("$z/[R_0]$")
    ax[1].set_xlabel("$y/[R_0]$")
    ax[1].set_ylabel("$z/[R_0]$")
    ax[0].set_title("$xz$-plane")
    ax[1].set_title("$yz$-plane")
    ax[0].set_xlim(-L, L)
    ax[0].set_ylim(-L, L)
    return ax

def plot1():
    ax = get_ax()
    legend_element = plot_field(ax)
    ax[0].legend(legend_element[2:], legends)
    ax[1].legend(legend_element[2:], legends)
    plt.savefig("figs/b_field_2D.png")

def plot2():
    for i in range(n_y):
        ax = get_ax()
        legend_element = plot_field(ax, alpha = 0.2)
        plot_lines(ax, i)
        plt.savefig("figs/charged_particles_{}_2D.png".format(i))

plot1()
plot2()