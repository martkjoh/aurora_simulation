from parametres import *
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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

def f(y):
    return np.array([y[1], np.cross(y[1], B(y[0]))])

def write_path(ys, i):
    shape = ys.shape    
    A = np.reshape(ys, (shape[0], shape[2] * 2))
    np.savetxt("data/particle_{}.csv".format(i), A, delimiter = ",")

def read_path(i):
    A = np.loadtxt("data/particle_{}.csv".format(i), delimiter = ",")
    shape = A.shape
    print(np.reshape(A, (shape[0], 2, 3)))

def simulate_paths():
    for i in range(len(X0s)):
        # y[i, j, k] is the k'th coordinat of the j'th derivative of x wrt t at t = dt*i
        ys = np.empty((N, 2, 3))
        y0 = np.array([X0s[i], XDot0])
        ys[0] = y0

        for j in range(N - 1):
            RK4(ys, f, j)
        
        write_path(ys, i)

simulate_paths()
read_path(0)