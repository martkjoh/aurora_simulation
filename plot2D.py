from parametres import *
from particles import read_path, B, dot


n = 20
L = 2
x = np.array(np.mgrid[-L:L:n*1j, -L:L:n*1j])
legends = ["$\\vec B/[B_0]$", "$\\hat m$",]

def energy(ys):
    xDot = ys[:, 1, :]
    xDot2 = np.einsum("ti, ti -> t", xDot, xDot)
    return 1 / 2 * xDot2
    
def energyError(E):
    E0 = E[0] * np.ones_like(E)
    return abs(E - E0) / E0

def mask2D(f, R):
    r = np.sqrt(dot(x, x))
    f = np.ma.array(f)
    return np.ma.array([np.ma.masked_where(r < R, f[i]) for i in range(2)])

def plot_field(ax, alpha = 1):
    # Getting the xz and yz componenes of the dp-moment
    ms = np.array([[m[0], m[2]], [m[1], m[2]]])
    origin = np.mgrid[0:1:1, 0:1:1]
    legend_element = []
    for i in range(2):
        mField = np.mgrid[ms[i, 0]:ms[i, 0]:n*1j, ms[i, 1]:ms[i, 1]:n*1j]
        Bx = B(x, mField)
        Bx = mask2D(Bx, 1)
        legend_element.append(ax[i].quiver(*x, *Bx, pivot = "middle", alpha = 0.9 *alpha))
        earth = plt.Circle((0, 0), 1, color = "blue", alpha = 0.2 * alpha)
        ax[i].add_artist(earth)
        legend_element.append(ax[i].quiver(*origin, *ms[i].T, scale = 2, pivot = "middle", color = "red", alpha = 0.8 * alpha))

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
    ax[0].set_xlabel("$x/[R_\\odot]$")
    ax[0].set_ylabel("$z/[R_\\odot]$")
    ax[1].set_xlabel("$y/[R_\\odot]$")
    ax[0].set_title("$xz$-plane")
    ax[1].set_title("$yz$-plane")
    ax[0].set_xlim(-L, L)
    return ax

def get_ax2():
    fig, ax = plt.subplots(n_y, 2, sharex = True, sharey = True, figsize = (20, 32))
    ax[-1][0].set_xlabel("$x/[R_\\odot]$")
    ax[0][0].set_ylabel("$z/[R_\\odot]$")
    ax[-1][1].set_xlabel("$y/[R_\\odot]$")
    ax[0][0].set_title("$xz$-plane")
    ax[0][1].set_title("$yz$-plane")
    ax[0][0].set_xlim(-L, L)
    ax[0][0].set_ylim(-L/4, L)
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

def get_ax4():
    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (20, 18))
    ax[1][0].set_xlabel("$x/[R_\\odot]$")
    ax[0][0].set_ylabel("$z/[R_\\odot]$")
    ax[1][1].set_xlabel("$y/[R_\\odot]$")
    ax[0][0].set_title("$xz$-plane")
    ax[0][1].set_title("$yz$-plane")
    ax[0][0].set_xlim(-1, 0)
    ax[0][0].set_ylim(0, 1)
    return ax

def plot1():
    ax = get_ax1()
    legend_element = plot_field(ax)
    ax[0].legend(legend_element[2:], legends)
    ax[1].legend(legend_element[2:], legends)
    plt.savefig("figs/b_field_2D.pdf")

def plot2():
    axs = get_ax2()
    for i in range(n_y):
        ax = axs[i]
        legend_element = plot_field(ax, alpha = 0.2)
        plot_lines(ax, i)

    plt.tight_layout()    
    plt.savefig("figs/charged_particles_2D.pdf")

def plot3():
    ax, fig = get_ax3()
    n = n_y * n_z
    t = np.linspace(0, T, N)
    maximum = [0, 0]
    for i in range(n):
        ys = read_path(i)
        E = energy(ys)
        dE = energyError(E)
        if max(dE) > maximum[0]:
            maximum[0] = max(dE)
            maximum[1] = i
        ax.plot(t, dE, color = cm.plasma(i / n))

    print("max relative error was {}, by particle {}".format(*maximum))
    plt.savefig("figs/relative_error_energy.pdf")

def plot4():
    axs = get_ax4()
    for i in range(2):
        ax = axs[i]
        legend_element = plot_field(ax, alpha = 0.2)
        plot_lines(ax, i)

    plt.tight_layout()    
    plt.savefig("figs/charged_particles_zoom.pdf")

def make_plots():
    plot1()
    plot2()
    plot3()
    plot4()