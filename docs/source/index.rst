pyADiff: A simple, pure python algorithmic differentiation package
==================================================================

`pyADiff` is a (yet) very basic algorithmic differentiation package, which implements forward and adjoint/reverse mode differentiation.
If you are looking for a fully-featured and faster library, have a look at `google/jax`_, `autograd`_ or `dco/c++`_ (or many more),  but if you are interested in a package where you are able to quickly "look under the hood", you may be right here.

.. _`google/jax`: https://github.com/google/jax
.. _`autograd`: https://github.com/HIPS/autograd
.. _`dco/c++`: https://www.stce.rwth-aachen.de/research/software/dco/cpp

Contents
-------- 
.. toctree::
    :glob:
    :maxdepth: 2

    documentation/examples
    documentation/code_doc


Basic Usage
-----------

Suppose we want to compute the gradient of the function :math:`f(x_0, x_1) = 2 x_0 x_1^2`. This is a rather trivial task, because by simple calculus the gradient is:

.. math::
    \nabla f(x_0, x_1) = \begin{pmatrix} 2 x_1^2 \\ 4 x_0 x_1\end{pmatrix}

Nevertheless we use this example illustrate the use of `pyADiff`.

.. code-block:: python

    import pyADiff as ad
    # define the function f
    def f(x):
        return 2.*x[0]*x[1]**2.
    # call the gradient function of pyADiff
    df = ad.gradient(f)

    x = [0.5, 2.0]
    # Call the function f and the gradient function df
    y = f(x)
    dy = df(x)

    print("f({}) = {}".format(x, y))  # prints f([0.5, 2.0]) = 4.0
    print("f'({}) = {}".format(x, dy))  # prints f'([0.5, 2.0]) = [8. 4.]

Which corresponds to the evaluation of the analytic gradient.

.. math::
    \nabla f(0.5, 2) = \begin{pmatrix} 2*2^2 \\ 4 * 0.5 * 2\end{pmatrix} = \begin{pmatrix} 8 \\ 4 \end{pmatrix}

Motivation
----------
My motivation to start this project arose from curiosity while listening to the lecture `Computational Differentiation`_ by `Uwe Naumann`_ at `RWTH Aachen University`_.
So basically I tried to understand the concepts from the lecture by implementing them by myself.
In the end I was (positively) surprised with the outcome and decided to bundle it in a python package.
Additionaly this gave me the chance to learn about python packaging, distributing, documentation, ...

.. _`Computational Differentiation`: https://www.stce.rwth-aachen.de/teaching/lectures/computational-differentiation
.. _`Uwe Naumann`: https://www.stce.rwth-aachen.de/people/uwe-naumann
.. _`RWTH Aachen University`: https://www.rwth-aachen.de/

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
