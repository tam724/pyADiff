"""Module Differentiation.

The functions defined here wrap the tangent and adjoint derivative computations and return them as lambda functions.

The tangent/forward computation is wrapped as `derfor`, the adjoint/reverse computation as `derrev`.
For convenience also the functions `derivative`, `gradient` and `hessian` are implemented, but they simply call `derfor`/`derrev`.
"""
import pyADiff
import pyADiff.tangent as pyADiff_tangent
import pyADiff.adjoint as pyADiff_adjoint


def derfor(f):
    """Forward Differentiation.

    Wraps the calculation of the derivative of `f` with respect to its inputs via tangent mode differentiation.
    The signature of `f` is assumed to be::

        {scalar, list, array} = f({scalar, list, array})

    Assuming the mathematical signature of :math:`f` to be

    .. math:: f: \\mathbb{R}^{i \\times j \\times ...} \\to \\mathbb{R}^{m \\times n \\times ...}

    the mathematical signature of the derivative :math:`f'` will be

    .. math:: f': \\mathbb{R}^{i \\times j \\times ...} \\to \\mathbb{R}^{m \\times n \\times ... \\times i \\times j \\times ...}

    Parameters
    ----------
    f : function_type
        The function to be differentiated. 

    Returns
    -------
    function_type
        A function which returns the derivative of f.

    See also
    --------
    pyADiff.tangent.dfdx : The function which actually computes the derivative.
    pyADiff.tangent.ADTypeT : The overloaded scalar ADType which is used for the computation.
    """
    return lambda x: pyADiff_tangent.dfdx(f, x)

def derrev(f):
    """Adjoint Differentiation.

    Wraps the calculation of the derivative of f with respect to its inputs via adjoint mode differentiation.
    The signature of f is assumed to be::

        {scalar, list, array} = f({scalar, list, array})

    Assuming the mathematical signature of :math:`f` to be

    .. math:: f: \\mathbb{R}^{i \\times j \\times ...} \\to \\mathbb{R}^{m \\times n \\times ...}

    the mathematical signature of the derivative :math:`f'` will be

    .. math:: f': \\mathbb{R}^{i \\times j \\times ...} \\to \\mathbb{R}^{m \\times n \\times ... \\times i \\times j \\times ...}

    Parameters
    ----------
    f : function_type
        The function to be differentiated. 

    Returns
    -------
    function_type
        A function which returns the derivative of f.

    See also
    --------
    pyADiff.adjoint.dfdx : Function actually computes the derivative.
    pyADiff.adjoint.ADTypeA : The overloaded scalar ADType which holds the derivtives.
    pyADiff.adjoint.ADRecord : The object which holds the "record" of single assignment operations.
    """
    return lambda x: pyADiff_adjoint.dfdx(f, x)

def derivative(f):
    """Derivative Computation

    Uses tangent mode differentiation to calculate the derivative.
    Mathematically one speaks of a derivative for functions :math:`f:\\mathbb{R} \\to \\mathbb{R}`, then the derivative is :math:`f': \\mathbb{R} \\to \\mathbb{R}`.
    
    Nevertheless `f` is not assumed to have this signature.

    Parameters
    ----------
    f : function_type
        The function to be differentiated. 

    Returns
    -------
    function_type
        A function which returns the derivative of f.

    See also
    --------
    pyADiff.differentiation.derfor : Wrapper for the comutation of the derivative via tangent mode.
    """
    return derfor(f)

def gradient(f):
    """Gradient Computation

    Uses adjoint mode differentiation to calculate the gradient.
    Mathematically one speaks of a gradient for functions :math:`f:\\mathbb{R}^n \\to \\mathbb{R}`, then the gradient is :math:`\\nabla f: \\mathbb{R}^n \\to \\mathbb{R}^n`.
    
    Nevertheless `f` is not assumed to have this signature.

    Parameters
    ----------
    f : function_type
        The function to be differentiated. 

    Returns
    -------
    function_type
        A function which returns the gradient of f.

    See also
    --------
    pyADiff.differentiation.derrev : Wrapper for the comutation of the derivative via adjoint mode.
    """
    return derrev(f)
    
def hessian(f):
    """Hessian Computation

    Uses tangent and adjoint mode differentiation to calculate the hessian.
    Mathematically one speaks of a hessian for functions :math:`f:\\mathbb{R}^n \\to \\mathbb{R}`, then the hessian is :math:`Hf: \\mathbb{R}^n \\to \\mathbb{R}^{n \\times n}`.
    
    Nevertheless `f` is not assumed to have this signature.

    Parameters
    ----------
    f : function_type
        The function to be differentiated. 

    Returns
    -------
    function_type
        A function which returns the hessian of f.

    See also
    --------
    pyADiff.differentiation.derfor : Wrapper for the comutation of the derivative via tangent mode.
    pyADiff.differentiation.derrev : Wrapper for the comutation of the derivative via adjoint mode.
    """
    return derfor(derrev(f))

