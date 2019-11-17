import pyADiff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize

def generalizedRosenbrock(x):
    n = len(x)
    y = 0.
    for i in range(0, n - 1):
        y += (1. - x[i])**2. + 100.*(x[i+1] - x[i]**2.)**2.
    return y

def plot():
    x = np.linspace(-2, 2, 1000)
    y = np.linspace(-1, 3, 1000)
    y, x = np.meshgrid(y, x)
    z = np.zeros(x.shape)
    for i in range(1000):
        for j in range(1000):
            z[i, j] = generalizedRosenbrock([x[i, j], y[i, j]])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z)
    # plt.contourf(x, y, z, levels=np.linspace(z.min(), z.max(), 100))
    plt.show()

def optimize():
    f = generalizedRosenbrock
    df = pyADiff.derrev(f)
    ddf = pyADiff.derfor(pyADiff.derrev(f))

    x = np.array([-0.5, 2.])
    x = np.random.random(20)
    epsilon = 0.0000001
    gradient = df(x)
    increment = np.zeros(x.shape)
    p = 0.3
    while np.linalg.norm(gradient) > epsilon:
        increment = (1 - p)*np.linalg.solve(ddf(x), gradient) + p*increment
        x -= increment
        gradient = df(x)
        print(np.linalg.norm(gradient))
    return x
print(optimize())