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

def energy(ys):
    xDot = ys[:, 1, :]
    xDot2 = np.einsum("ti, ti -> t", xDot, xDot)
    return 1 / 2 * xDot2
def deltaE(E):
    E0 = E[0] * np.ones_like(E)
    return abs(E - E0) / E0


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

def get_ax1():
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

def get_ax2():
    fig, ax = plt.subplots(n_y, 2, sharex = True, sharey = True, figsize = (20, 32))
    ax[-1][0].set_xlabel("$x/[R_0]$")
    ax[0][0].set_ylabel("$z/[R_0]$")
    ax[-1][1].set_xlabel("$y/[R_0]$")
    ax[0][0].set_title("$xz$-plane")
    ax[0][1].set_title("$yz$-plane")
    ax[0][0].set_xlim(-L, L / 4)
    ax[0][0].set_ylim(-L/4, L)
    ax[0][1].set_xlim(-L, L)
    ax[0][1].set_ylim(-L/4, L)
    return ax

def get_ax3():
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.set_yscale("log")
    ax.set_ylabel("$\Delta E$")
    ax.set_xlabel("$t / [t_0]$")
    n = n_y * n_z
    x = np.linspace(0, n, n + 1)
    cmap = plt.get_cmap("plasma", n)
    norm = mpl.colors.BoundaryNorm(x - 0.5, n)
    sm = plt.cm.ScalarMappable(norm = norm, cmap=cmap)
    sm.set_array([]) 
    plt.colorbar(sm, ticks = x, label = "particle #")
    return ax, fig

def plot1():
    ax = get_ax1()
    legend_element = plot_field(ax)
    ax[0].legend(legend_element[2:], legends)
    ax[1].legend(legend_element[2:], legends)
    plt.savefig("figs/b_field_2D.png")

def plot2():
    axs = get_ax2()
    for i in range(n_y):
        ax = axs[i]
        legend_element = plot_field(ax, alpha = 0.2)
        plot_lines(ax, i)

    plt.tight_layout()    
    plt.savefig("figs/charged_particles_2D.png")

def plot3():
    ax, fig = get_ax3()
    n = n_y * n_z
    t = np.linspace(0, T, N)
    maximum = [0, 0]
    for i in range(n):
        ys = read_path(i)
        E = energy(ys)
        dE = deltaE(E)
        if max(dE) > maximum[0]:
            maximum[0] = max(dE)
            maximum[1] = i
        ax.plot(t, dE, color = cm.plasma(i / n))

    print("max relative error was {}, by particle {}".format(*maximum))
    plt.savefig("figs/relative_error_energy.png")

def make_plots():
    plot1()
    plot2()
    plot3()