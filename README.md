# pyADiff: A simple, pure python algorithmic differentiation package

`pyADiff` is a (yet) very basic algorithmic differentiation package, which implements forward and adjoint/reverse mode differentiation. If you are looking for a fully-featured and faster library, have a look at [google/jax](https://github.com/google/jax), [autograd](https://github.com/HIPS/autograd) or [dco/c++](https://www.stce.rwth-aachen.de/research/software/dco/cpp) (or many more),  but if you are interested in a package where you are able to quickly "look under the hood", you may be right here.

## Motivation
My motivation to start this project arose from curiosity while listening to the lecture ["Computational Differentiation"](https://www.stce.rwth-aachen.de/teaching/lectures/computational-differentiation) by [Uwe Naumann](https://www.stce.rwth-aachen.de/people/uwe-naumann) at [RWTH Aachen University](https://www.rwth-aachen.de/). So basically I tried to understand the concepts from the lecture by implementing them by myself. In the end I was (positively) surprised with the outcome and decided to bundle it in a python package. Additionaly this gave me the chance to learn about python packaging, distributing, documentation, ...

## Basic Usage
Suppose we want to compute the gradient of the function $f(x_0, x_1) = 2 x_0 x_1^2$. This is a rather trivial task, because by simple calculus, the gradient is:
$$
\nabla f(x_0, x_1) = \begin{pmatrix} 2 x_1^2 \\ 4 x_0 x_1\end{pmatrix}
$$
Nevertheless we use this example illustrate the use of `pyADiff`.
```python
import pyADiff as ad
def f(x):
	return 2.*x[0]*x[1]**2.
df = ad.gradient(f)

x = [0.5, 2.0]
y = f(x)
dy = df(x)
print("f({}) = {}".format(x, y)) 
# prints f([0.5, 2.0]) = 4.0
print("f'({}) = {}".format(x, dy))
# prints f'([0.5, 2.0]) = [8. 4.]
```
Which corresponds to the evaluation of the analytic gradient at $(x_0, x_1) = (0.5, 2)$.

For more sophisticated examples see the [Documentation](#documentation) or have a look at the [.ipynb notebooks](/docs/source/documentation/examples)

## Installation
### Installation using pip
*TODO*

### Installation from source
This will clone the repository and install the `pyADiff` package using the `setup.py` script.
```shell
> git clone https://github.com/tam724/pyADiff
> python pyADiff/setup.py install
```

## Documentation
*TODO*

## References
### Algorithmic Differentiation:
* Uwe Naumann, *Lecture Computational Differentiation*, RWTH Aachen