import numpy as np

def RK4(y, f, dt, i, *args):
    k1 = f(y[i], *args) * dt
    k2 = f(y[i] + k1 / 2, *args) * dt
    k3 = f(y[i] + k2 / 2, *args) * dt
    k4 = f(y[i] + k3, *args) * dt
    y[i + 1]  = y[i] + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Particle:
    def __init__(self, r, v):
        self.r = r
        self.v = v

    def update(F, dt)

print(pi)