import numpy as np

from context import pyADiff

sin = pyADiff.sin
cos = pyADiff.cos
exp = pyADiff.exp
log = pyADiff.log

def test_derivative_f1():
    def f(x):
        return sin(x[0])*x[1] - x[0]
    def df_analytic(x):
        return np.array([
            - 1. + cos(x[0])*x[1],
            sin(x[0])
        ])
    df_for = pyADiff.derfor(f)
    df_rev = pyADiff.derrev(f)

    x = np.array([1., 3.])
    assert(np.all(np.isclose(df_analytic(x), df_for(x))))
    assert(np.all(np.isclose(df_analytic(x), df_rev(x))))

    x = np.array([10., 0.5])
    assert(np.all(np.isclose(df_analytic(x), df_for(x))))
    assert(np.all(np.isclose(df_analytic(x), df_rev(x))))

    def ddf_analytic(x):
        return np.array([
            [-sin(x[0])*x[1], cos(x[0])],
            [cos(x[0]), 0.]
        ])
    
    ddf_for_for = pyADiff.derfor(pyADiff.derfor(f))
    ddf_for_rev = pyADiff.derfor(pyADiff.derrev(f))
    ddf_rev_for = pyADiff.derrev(pyADiff.derfor(f))
    ddf_rev_rev = pyADiff.derrev(pyADiff.derrev(f))
    
    x = np.array([1., 3.])
    assert(np.all(np.isclose(ddf_analytic(x), ddf_for_for(x))))
    assert(np.all(np.isclose(ddf_analytic(x), ddf_for_rev(x))))
    assert(np.all(np.isclose(ddf_analytic(x), ddf_rev_for(x))))
    assert(np.all(np.isclose(ddf_analytic(x), ddf_rev_rev(x))))

def test_derivative_f2():
    def f(x):
        return x[1]*x[2]/x[0]
    def df_analytic(x):
        return np.array([
            -x[1]*x[2]/(x[0]*x[0]),
            x[2]/x[0],
            x[1]/x[0]
        ])
    df_for = pyADiff.derfor(f)
    df_rev = pyADiff.derrev(f)

    x = np.array([0.5, 7., -2.])
    assert(np.all(np.isclose(df_analytic(x), df_for(x))))
    assert(np.all(np.isclose(df_analytic(x), df_rev(x))))

    x = np.array([-0.4, 3.45, 9.8])
    assert(np.all(np.isclose(df_analytic(x), df_for(x))))
    assert(np.all(np.isclose(df_analytic(x), df_rev(x))))

def test_derivative_f3():
    def f(x):
        return x[0]**x[1]
    def df_analytic(x):
        if x[0] == 0.:
            return np.array([
                x[1]*x[0]**(x[1] - 1.),
                0.  
            ])
        else:
            return np.array([
                x[1]*x[0]**(x[1] - 1.),
                x[0]**x[1]*log(x[0])
            ])
    df_for = pyADiff.derfor(f)
    df_rev = pyADiff.derrev(f)

    x = np.array([0.5, 7.])
    assert(np.all(np.isclose(df_analytic(x), df_for(x))))
    assert(np.all(np.isclose(df_analytic(x), df_rev(x))))

    x = np.array([10., 9.8])
    assert(np.all(np.isclose(df_analytic(x), df_for(x))))
    assert(np.all(np.isclose(df_analytic(x), df_rev(x))))

def test_derivative_f4():
    def f(x):
        return 3 * exp(2.**x[0]) - (x[1]/x[2])**4. * sin(x[2]*x[1]*exp(x[0])**2.)
    def df_analytic(x):
        return np.array([
            (-2.*exp(2*x[0])*x[1]**5.*cos(exp(2.*x[0])*x[1]*x[2]))/x[2]**3. + 2.**x[0]*exp(2.**x[0])*log(8.),
            -((x[1]**3.*(exp(2.*x[0])*x[1]*x[2]*cos(exp(2.*x[0])*x[1]*x[2]) + 4.*sin(exp(2.*x[0])*x[1]*x[2])))/x[2]**4.),
            -((x[1]**4.*(exp(2.*x[0])*x[1]*x[2]*cos(exp(2.*x[0])*x[1]*x[2]) - 4.*sin(exp(2.*x[0])*x[1]*x[2])))/x[2]**5.)
        ])
    df_for = pyADiff.derfor(f)
    df_rev = pyADiff.derrev(f)

    x = np.array([3., 7., 10.])
    assert(np.all(np.isclose(df_analytic(x), df_for(x))))
    assert(np.all(np.isclose(df_analytic(x), df_rev(x))))

    x = np.array([-5., 4., 9.])
    assert(np.all(np.isclose(df_analytic(x), df_for(x))))
    assert(np.all(np.isclose(df_analytic(x), df_rev(x))))