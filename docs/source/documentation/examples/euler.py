import pyADiff
import numpy as np
import matplotlib.pyplot as plt

def explicit_euler(x0, n, dt, f):
    t = np.zeros(n)
    x = np.zeros(n, dtype=type(x0))
    x[0] = x0
    for i in range(1, n):
        x[i] = x[i-1] + dt*f(x[i-1], t)
        t[i] = t[i-1] + dt
    return t, x

def f(x, t, nu):
    return -nu*x

dt = 0.01
n = 100

x0_true = 10.
nu_true = 3.

t_true, x_true = explicit_euler(x0_true, n, dt, lambda x, t: f(x, t, nu_true))

measurement_indices = [20, 50, 80]
measurenments = [x_true[i] for i in measurement_indices]

def cost_function(p):
    x0 = p[0]
    nu = p[1]
    t, x = explicit_euler(x0, n, dt, lambda x, t: f(x, t, nu))
    sim_measurenments = [x[i] for i in measurement_indices]
    squared_error = 0.
    for i in range(len(measurenments)):
        squared_error += (measurenments[i] - sim_measurenments[i])**2.
    return squared_error

def minimize(p0, a, f, epsilon):
    p = p0
    df = pyADiff.gradient(f)

    gradient = df(p)
    save = []
    save.append({
        'x0': p[0],
        'nu': p[1],
        'norm': np.linalg.norm(gradient)
        })
    while np.linalg.norm(gradient) > epsilon:
        p -= a*gradient
        gradient = df(p)
        save.append({
            'x0': p[0],
            'nu': p[1],
            'norm': np.linalg.norm(gradient)
            })
        print(np.linalg.norm(gradient))
        print(cost_function(p))
    return p, save


p_opt, save = minimize(np.array([1., 1.]), 0.3, cost_function, 0.1)
print("The optimum was found at (x0, nu) = ({}, {})".format(p_opt[0], p_opt[1]))
print("The truth was (x0, nu) = ({}, {})".format(x0_true, nu_true))

plt.plot(t_true, x_true, label="true")
for i in measurement_indices:
    plt.plot(t_true[i], x_true[i], 'o')

for i, xx in enumerate(save):
    x0 = xx['x0']
    nu = xx['nu']
    x, t = explicit_euler(x0, n, dt, lambda x, t: f(x, t, nu))
    plt.plot(t, x, label="step {}".format(i))
plt.legend()

plt.figure()
norm_evolution = [xx['norm'] for xx in save]
x0_evolution = [xx['x0'] for xx in save]
nu_evolution = [xx['nu'] for xx in save]
plt.plot(norm_evolution, label="norm of the gradient", color="C1")
plt.axhline(x0_true, linestyle=':', label="true x0", color="C2")
plt.plot(x0_evolution, label="x0", color="C2")
plt.axhline(nu_true, linestyle=':', label="true nu", color="C3")
plt.plot(nu_evolution, label="nu", color="C3")
plt.legend()

plt.show()