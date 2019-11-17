"""Module Mathematical Functions

Collection of the suppported, overloaded mathematical functions (e.g. sin, cos, exp, ..).

The ADTypes implement these functions as member functions.
So in case the input `a` is an ADType, calling the respective member returns a new ADType with the calculated value (and its derivative representation).
If the input `a` does not implement the mathematical function, a `AttributeError` is raised and we fall back to the `numpy` implementation.

See also
--------
pyADiff.tangent.ADTypeT: Implementation of the tangent ADType.
pyADiff.adjoint.ADTypeA: Implementation of the adjoint ADType.
"""
import numpy as np

import pyADiff

def sin(a):
    """Sine.

    Implements the overload of the sine function.

    Parameters
    ----------
    a : scalar, array of float or ADType
        Input of the function.
    """
    try:
        return a.sin()
    except AttributeError:
        return np.sin(a)

def cos(a):
    """Cosine.

    Implements the overload of the cosine function.

    Parameters
    ----------
    a : scalar, array of float or ADType
        Input of the function.
    """
    try:
        return a.cos()
    except AttributeError:
        return np.cos(a)

def exp(a):
    """Exponential.

    Implements the overload of the exponential function.

    Parameters
    ----------
    a : scalar, array of float or ADType
        Input of the function.
    """
    try:
        return a.exp()
    except AttributeError:
        return np.exp(a)

def log(a):
    """Logarithm.

    Implements the overload of the logarithm function.

    Parameters
    ----------
    a : scalar, array of float or ADType
        Input of the function.
    """
    try:
        return a.log()
    except AttributeError:
        return np.log(a)

def sqrt(a):
    """Square root.

    Implements the overload of the square root function.

    Parameters
    ----------
    a : scalar, array of float or ADType
        Input of the function.
    """
    try:
        return a.sqrt()
    except AttributeError:
        return np.sqrt(a)
