from mayavi import mlab
from parametres import *
from particles import read_path

x = np.array((np.mgrid[
    -L:L:dims[0]*1j, 
    -L:L:dims[1]*1j, 
    -L:L:dims[2]*1j
    ]))

def dot(f1, f2):
    return np.einsum("ixyz,ixyz->xyz", f1, f2)

def f(x):
    r = np.sqrt(dot(x, x))
    return sin(x) / x

mlab.contour3d(*x, f(x))5