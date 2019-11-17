import numpy as np

import pyADiff
from pyADiff.exceptions import NotDifferentiableExeption
from pyADiff.math_functions import *


class ADRecord(object):
    """ADRecord.

    Stores all operations of a computation in a list, preserving their order.
    
    See also
    --------
    pyADiff.adjoint.ADTypeA : The overloaded adjoint numerical type.
    """
    def __init__(self):
        self._record = []
    
    def record_variable(self, v):
        """Adds a variable to the record.
        
        Parameters
        ----------
        v : ADTypeA
            Variable to record.
        """
        self._record.append(v)

    def backpropagate(self):
        """Backpgropagation.

        Backpropagates through all stored computations in reversed order.
        """
        for v in reversed(self._record):
            v.backpropagate()

    def reset(self):
        """Resets the derivatives.

        Resets the derivatives of all values to 0.
        """
        for v in self._record:
            v.derivative = 0.

class ADTypeA(object):
    """Adjoint ADType.

    This class overloads the basic numerical type of python.
    Instead of only an `value` it also stores a `derivative`, its `dependencies` and a `record`.

    This class implements the numerical operators (+, -, .. ) as expected for a numerical type, but additionaly stores the operands (`dependencies`) and records the operation to the `record`.
    
    Basic mathematical functions (`sin`, `cos`, `exp`, ...) are implemented as member functions and also store the operand in the return `ADTypeA` and record the operation.

    Parameters
    ----------
    value : float or ADType
        The value of the overloaded numerical type.
    record : ADRecord
        The record of all operations.
    dependencies : list[tuple(ADTypeA, float or ADType)]
        List of the ADTypeAs this ADTypeA depends on and their partial derivatives.
    deriative : float or ADType
        The derivative of the overloaded numerical type.

    See also
    --------
    pyADiff.adjoint.ADRecord : Records all operations.
    pyADiff.math_functions : Implementation of basic mathematical functions for the ADType.
    """
    def __init__(self, value, record, dependencies=[], derivative=0.):
        self._v = value
        self._d = derivative
        self._deps = dependencies
        self._r = record
        self._r.record_variable(self)
            
    @property
    def value(self):
        """Value of the overloaded numerical type.
        """
        return self._v
    
    @value.setter
    def value(self, value):
        self._v = value

    @property
    def derivative(self):
        """Derivative of the overloaded numerical type.
        """
        return self._d

    @derivative.setter
    def derivative(self, derivative):
        self._d = derivative

    @property
    def dependencies(self):
        """Dependencies of the ADTypeA

        List of tuples(ADTypeA, float or ADType) which represent the dependencies of this ADTypeA.
        """
        return self._deps

    def backpropagate(self):
        """Backpropagation.

        Calculates and accumulates the partial derivatives of its dependencies.
        """
        for v, p in self.dependencies:
            v.derivative += p*self.derivative

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __add__(self, other):
        try:
            return ADTypeA(
                value=self.value + other.value,
                record=self._r,
                dependencies=[
                    (self, 1.),
                    (other, 1.)
                ]
            )
        except AttributeError:
            temp = other.__radd__(self)
            if temp is NotImplemented:
                return ADTypeA(
                    value=self.value + other,
                    record=self._r,
                    dependencies=[
                        (self, 1.),
                    ]
                )
            else:
                return temp

    def __radd__(self, other):
        return ADTypeA(
            value=self.value + other,
            record=self._r,
            dependencies=[
                (self, 1.)
            ]
        )

    def __sub__(self, other):
        try:
            return ADTypeA(
                value=self.value - other.value,
                record=self._r,
                dependencies=[
                    (self, 1.),
                    (other, -1.)
                ]
            )
        except AttributeError:
            temp = other.__rsub__(self)
            if temp is NotImplemented:
                return ADTypeA(
                    value=self.value - other,
                    record=self._r,
                    dependencies=[
                        (self, 1.),
                    ]
                )
            else:
                return temp

    def __rsub__(self, other):
        return ADTypeA(
            value=other - self.value,
            record=self._r,
            dependencies=[
                (self, -1.)
            ]
        )

    def __mul__(self, other):
        try:
            return ADTypeA(
                value=self.value * other.value,
                record=self._r,
                dependencies=[
                    (self, other.value),
                    (other, self.value)
                ]
            )
        except AttributeError:
            temp = other.__rmul__(self)
            if temp is NotImplemented:
                return ADTypeA(
                    value=self.value * other,
                    record=self._r,
                    dependencies=[
                        (self, other),
                    ]
                )
            else:
                return temp

    def __rmul__(self, other):
        return ADTypeA(
            value=self.value*other,
            record=self._r,
            dependencies=[
                (self, other)
            ]
        )

    def __truediv__(self, other):
        try:
            return ADTypeA(
                value=self.value/other.value,
                record=self._r,
                dependencies=[
                    (self, 1./other.value),
                    (other, -self.value/other.value**2.)
                ]
            )
        except AttributeError:
            temp = other.__rtruediv__(self)
            if temp is NotImplemented:
                return ADTypeA(
                    value=self.value/other,
                    record=self._r,
                    dependencies=[
                        (self, 1./other),
                    ]
                )
            else:
                return temp
    
    def __rtruediv__(self, other):
        return ADTypeA(
            value=other/self.value,
            record=self._r,
            dependencies=[
                (self, -other/self.value**2.)
            ]
        )

    def __pow__(self, other):
        try:
            return ADTypeA(
                value=self.value**other.value,
                record=self._r,
                dependencies=[
                    (self, other.value*self.value**(other.value-1.)),
                    (other, self.value**other.value*log(self.value))
                ]
            )
        except AttributeError:
            temp = other.__rpow__(self)
            if temp is NotImplemented:
                return ADTypeA(
                    value=self.value**other,
                    record=self._r,
                    dependencies=[
                        (self, other*self.value**(other-1.)),
                    ]
                )
            else:
                return temp

    def __rpow__(self, other):
        return ADTypeA(
            value=other**self.value,
            record=self._r,
            dependencies=[
                (self, other**self.value*log(other))
            ]
        )

    ### SINGLE INPUT FUNCTIONS
    def __neg__(self):
        return ADTypeA(
            value=-self.value,
            record=self._r,
            dependencies=[
                (self, -1.)
            ]
        )

    def __pos__(self):
        return ADTypeA(
            value=self.value,
            record=self._r,
            dependencies=(self, 1.)
        )

    def __abs__(self):
        raise NotImplementedError

    def sin(self):
        return ADTypeA(
            value=sin(self.value),
            record=self._r,
            dependencies=[
                (self, cos(self.value))
            ]
        )

    def cos(self):
        return ADTypeA(
            value=cos(self.value),
            record=self._r,
            dependencies=[
                (self, -sin(self.value))
            ]
        )

    def exp(self):
        return ADTypeA(
            value=exp(self.value),
            record=self._r,
            dependencies=[
                (self, exp(self.value))
            ]
        )
    
    def log(self):
        return ADTypeA(
            value=log(self.value),
            record=self._r,
            dependencies=[
                (self, 1./self.value)
            ]
        )

    def sqrt(self):
        return ADTypeA(
            value=sqrt(self.value),
            record=self._r,
            dependencies=[
                (self, 1./(2.*sqrt(self.value)))
            ]
        )

    def __hash__(self):
        return int(id(self))

def dfdx(f, x_v):
    """Adjoint Differentiation Driver.

    This computes the derivative of `f` with respect to `x` at the position `x_v`.
    The signature of `f` is assumed to be::

        {scalar, list, array} = f({scalar, list, array})

    This function converts the inputs x_v to their respective `ADTypeA`, runs the function `f` and simultaneously creates the record of all operations.
    Then it successively sets the derivtive of the outputs `y=f(x)` to one and backpropgates through the record.
    Then the derivative value can be collected from the inputs `x`.

    Only one forward run of `f` is necessary, no matter the dimension of `x`.
    The backpropagation is performed once for each output `y`.

    Parameters
    ----------
    f : function_type
        The function to differentiate.
    x_v : scalar, list, array
        The value where to evaluate the derivative.

    See also
    --------
    pyADiff.differentiation.derrev : Wrapper for the comutation of the derivative via adjoint mode.
    """
    rec = ADRecord()
    if(type(x_v) is np.ndarray):
        x = np.empty(x_v.shape, dtype=ADTypeA)
        for i in np.ndindex(x_v.shape):
            x[i] = ADTypeA(x_v[i], rec)
        y = f(x)
        if(type(y) is np.ndarray):
            df = None
            for j in np.ndindex(y.shape):
                y[j].derivative = 1.
                rec.backpropagate()
                if df is None:
                    df = np.empty(y.shape + x.shape, dtype=type(x.flat[0].derivative))
                for i in np.ndindex(x.shape):
                    df[j + i] = x[i].derivative
                y[j].derivative = 0.
                rec.reset()
        else:
            y.derivative = 1.
            rec.backpropagate()
            df = np.empty(x.shape, dtype=type(x.flat[0].derivative))
            for i in np.ndindex(x.shape):
                df[i] = x[i].derivative
            y.derivative = 0.
            rec.reset()
        return df
    elif(type(x_v) is list):
        return dfdx(f, np.array(x_v))
    else:
        x = ADTypeA(x_v, rec)
        y = f(x)
        if(type(y) is np.ndarray):
            df = None
            for j in np.ndindex(y.shape):
                y[j].derivative = 1.
                rec.backpropagate()
                if df is None:
                    df = np.empty(y.shape, dtype=type(x.derivative))
                df[j] = x.derivative
                y[j].derivative = 0.
                rec.reset()
        else:
            y.derivative = 1.
            rec.backpropagate()
            df = x.derivative
            y.derivative = 0.
            rec.reset()
        return df