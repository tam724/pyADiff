import numpy as np

import pyADiff
from pyADiff.exceptions import NotDifferentiableExeption
from pyADiff.math_functions import *


class ADTypeT(object):
    """Tangent ADType.

    This class overloads the basic numerical type of python.
    Instead of only an `value` it also stores a `derivative`.

    This class implements the numerical operators (+, -, .. ) as expected for a numerical type, but additionaly accumulates the partial derivatives.

    Basic mathematical functions (`sin`, `cos`, `exp`, ...) are implemented as member functions and also accumulate the partial derivatives.

    Parameters
    ----------
    value : float or ADType
        The value of the overloaded numerical type.
    deriative : float or ADType
        The derivative of the overloaded numerical type.

    See also
    --------
    pyADiff.math_functions: Implementation of basic mathematical functions for the ADType.
    """
    def __init__(self, value, derivative=0.):
        self._v = value
        self._d = derivative
    
    @property
    def value(self):
        """Value of the overloaded numerical type.
        """
        return self._v
    
    @value.setter
    def value(self, v):
        self._v = v

    @property
    def derivative(self):
        """Derivative of the overloaded numerical type.
        """
        return self._d

    @derivative.setter
    def derivative(self, d):
        self._d = d

    def __repr__(self):
        return str(self.value) + '+d' + str(self.derivative)

    def __str__(self):
        return str(self.value)

    def __add__(self, other):
        try:
            return ADTypeT(
                value=self.value + other.value,
                derivative=self.derivative + other.derivative
                )
        except AttributeError:
            temp = other.__radd__(self)
            if temp is NotImplemented:
                return ADTypeT(
                    value=self.value + other,
                    derivative=self.derivative
                )
            else:
                return temp
        
    def __radd__(self, other):
        return ADTypeT(
            value=self.value + other,
            derivative=self.derivative
            )

    def __sub__(self, other):
        try:
            return ADTypeT(
                value=self.value - other.value,
                derivative=self.derivative - other.derivative
                )
        except AttributeError:
            temp = other.__rsub__(self)
            if temp is NotImplemented:
                return ADTypeT(
                    value=self.value - other,
                    derivative=self.derivative
                )
            else:
                return temp

    def __rsub__(self, other):
        return ADTypeT(
            value = other - self.value,
            derivative=-self.derivative
        )

    def __mul__(self, other):
        try:
            return ADTypeT(
                value=self.value*other.value,
                derivative=other.value*self.derivative + self.value*other.derivative
            )
        except AttributeError:
            temp = other.__rmul__(self)
            if temp is NotImplemented:
                return ADTypeT(
                    value=self.value*other,
                    derivative=other*self.derivative
                )
            else:
                return temp

    def __rmul__(self, other):
        return ADTypeT(
            value=self.value*other,
            derivative=other*self.derivative
        )

    def __truediv__(self, other):
        try:
            return ADTypeT(
                value=self.value/other.value,
                derivative=self.derivative/other.value - self.value/other.value**2.*other.derivative
            )
        except AttributeError:
            temp = other.__rtruediv__(self)
            if temp is NotImplemented:
                return ADTypeT(
                    value=self.value/other,
                    derivative=self.derivative/other
                )
            else:
                return temp
    
    def __rtruediv__(self, other):
        return ADTypeT(
            value=other/self.value,
            derivative=-other/self.value**2.*self.derivative
        )

    def __pow__(self, other):
        try:
            return ADTypeT(
                value=self.value**other.value,
                derivative=other.value*self.value**(other.value-1.)*self.derivative + self.value**other.value*log(self.value)*other.derivative
            )
        except AttributeError:
            temp = other.__rpow__(self)
            if temp is NotImplemented:
                return ADTypeT(
                    value=self.value**other,
                    derivative=other*self.value**(other-1.)*self.derivative
                    )
            else:
                return temp

    def __rpow__(self, other):
        return ADTypeT(
            value=other**self.value,
            derivative=other**self.value*log(other)*self.derivative
        )

    ### EQUALITIES AND INEQUALITIES
    def __lt__(self, other):
        try:
            return self.value < other.value
        except AttributeError:
            return self.value < other

    def __le__(self, other):
        try:
            return self.value <= other.value
        except AttributeError:
            return self.value <= other
    
    def __eq__(self, other):
        try:
            return self.value == other.value
        except AttributeError:
            return self.value == other

    def __ne__(self, other):
        try:
            return self.value != other.value
        except AttributeError:
            return self.value != other

    def __gt__(self, other):
        try:
            return self.value > other.value
        except AttributeError:
            return self.value > other

    def __ge__(self, other):
        try:
            return self.value >= other.value
        except AttributeError:
            return self.value >= other

    ### SINGLE INPUT FUNCTIONS
    def __neg__(self):
        return ADTypeT(
            value=-self.value,
            derivative=-self.derivative
        )

    def __pos__(self):
        return ADTypeT(
            value=self.value,
            derivative=self.derivative
        )

    def __abs__(self):
        if self.value == 0 and self.derivative != 0:
            raise NotDifferentiableExeption
        return ADTypeT(
            value=abs(self.value),
            derivative=self.value/abs(self.value)*self.derivative
        )

    def sin(self):
        return ADTypeT(
            value=sin(self.value),
            derivative=cos(self.value)*self.derivative
            )

    def cos(self):
        return ADTypeT(
            value=cos(self.value),
            derivative=-sin(self.value)*self.derivative
            )
            
    def exp(self):
        return ADTypeT(
            value=exp(self.value),
            derivative=exp(self.value)*self.derivative
            )

    def log(self):
        return ADTypeT(
            value=log(self.value),
            derivative=self.derivative/self.value
        )

    def sqrt(self):
        return ADTypeT(
            value=sqrt(self.value),
            derivative=self.derivative/(2.*sqrt(self.value))
        )


def dfdx(f, x_v):
    """Tangent Differentiation Driver.

    This computes the derivative of `f` with respect to `x` at the position `x_v`.
    The signature of `f` is assumed to be::

        {scalar, list, array} = f({scalar, list, array})

    This function converts the inputs x_v to their respective `ADTypeT` and successively sets their derivative to 1, runs the function `f` and collects the derivative values from the outputs `y = f(x)`.
    If `x` is scalar only one forward run of `f` is necessary.

    Parameters
    ----------
    f : function_type
        The function to differentiate.
    x_v : scalar, list, array
        The value where to evaluate the derivative.

    See also
    --------
    pyADiff.differentiation.derfor : Wrapper for the comutation of the derivative via tangent mode.
    """
    if(type(x_v) is np.ndarray):
        x = np.empty(x_v.shape, dtype=ADTypeT)
        for i in np.ndindex(x_v.shape):
            x[i] = ADTypeT(x_v[i])
        df = None
        for i in np.ndindex(x.shape):
            x[i].derivative = 1.
            y = f(x)
            if(df is None):
                if(type(y) is np.ndarray):
                    df = np.empty(y.shape + x.shape, dtype=type(y.flat[0].derivative))
                else:
                    df = np.empty(x.shape, dtype=type(y.derivative))
            if(type(y) is np.ndarray):
                for j in np.ndindex(y.shape):
                    df[j+i] = y[j].derivative
            else:
                df[i] = y.derivative
            x[i].derivative = 0.
        return df
    elif(type(x_v) is list):
        return dfdx(f, np.array(x_v))
    else:
        x = ADTypeT(x_v)
        x.derivative = 1.
        y = f(x)
        if(type(y) is np.ndarray):
            df = np.empty(y.shape, dtype=type(y.flat[0].derivative))
            for j in np.ndindex(y.shape):
                df[j] = y[j].derivative
        else:
            df = y.derivative
        x.derivative = 0.
        return df