import numpy as np

from context import pyADiff

def test_numpy_assignment():
    def f(x):
        y = np.zeros(10, dtype=x.dtype)
        y[0] = x[0]
        for i in range(1, 10):
            y[i] = y[i-1]*x[i]
        return y

    def df_analytic(x):
        dy = np.zeros((10, 10))
        dy[0, 0] = 1.
        for i in range(1, 10):
            for j in range(0, i):
                dy[i, j] = dy[i-1, j]*x[i]
            dy[i, i] = dy[i-1, i-1]*x[i-1]
        return dy
        
    
    df_t = pyADiff.derfor(f)
    df_a = pyADiff.derrev(f)
    x = np.random.random(10)

    dy_t = df_t(x)
    dy_a = df_a(x)
    dy_analytic = df_analytic(x)
    
    assert(np.all(np.isclose(dy_t, dy_analytic)))
    assert(np.all(np.isclose(dy_a, dy_analytic)))