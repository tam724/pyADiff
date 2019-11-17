import numpy as np

from context import pyADiff

ADTypeT = pyADiff.tangent.ADTypeT

"""
BASIC OPERATOR TESTING
----------------------

This sections tests the basic operators of the pyAD tangent type.
In particular it ensures that the ADTypeT behaves as expectet in combination with numpy arrays.

All operators ~ (commutative or not) are tested in both ways:
 a ~ b and
 b ~ a

Table which describes the names:

b     ~    a|base       ad      numpy base  numpy ad
____________|___________________________________________
base        |X          bs_ad   X           npad_bs    
ad          |bs_ad      ad_ad   np_ad       npad_ad 
numpy base  |X          np_ad   X           npad_np
numpy ad    |npad_bs    npad_ad npad_np     npad_npad

"""


"""
SUM TESTS
---------

"""
def test_sum_bs_ad():
    x_bs = 1.
    x_ad = ADTypeT(1.)

    y = x_bs + x_ad
    assert(type(y) is ADTypeT)
    assert(y.value == 2.)
    
    y = x_ad + x_bs
    assert(type(y) is ADTypeT)
    assert(y.value == 2.)

def test_sum_ad_ad():
    x_ad1 = ADTypeT(1.)
    x_ad2 = ADTypeT(1.)

    y = x_ad1 + x_ad2
    assert(type(y) is ADTypeT)
    assert(y.value == 2.)

def test_sum_np_ad():
    x_np = np.arange(4, dtype=np.float)
    x_ad = ADTypeT(1.)

    y = x_np + x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == x_np[i] + 1.)
    
    y = x_ad + x_np
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == x_np[i] + 1.)

def test_sum_npad_bs():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i))
    x_bs = 1.

    y = x_np_ad + x_bs
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) + 1.)

    y = x_bs + x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) + 1.)

def test_sum_npad_ad():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i))
    x_ad = ADTypeT(1.)

    y = x_np_ad + x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) + 1.)

    y = x_ad + x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) + 1.)

def test_sum_npad_np():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i))
    x_np = np.arange(4, dtype=np.float)

    y = x_np_ad + x_np
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) + float(i))

    y = x_np + x_np_ad
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) + float(i))

def test_sum_npad_npad():
    x_np_ad1 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad1[i] = ADTypeT(float(i))
    x_np_ad2 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad2[i] = ADTypeT(float(i))

    y = x_np_ad1 + x_np_ad2
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) + float(i))

"""
SUB TESTS
---------
"""
def test_sub_bs_ad():
    x_bs = 1.
    x_ad = ADTypeT(1.)

    y = x_bs - x_ad
    assert(type(y) is ADTypeT)
    assert(y.value == 0.)
    
    y = x_ad - x_bs
    assert(type(y) is ADTypeT)
    assert(y.value == 0.)

def test_sub_ad_ad():
    x_ad1 = ADTypeT(1.)
    x_ad2 = ADTypeT(1.)

    y = x_ad1 - x_ad2
    assert(type(y) is ADTypeT)
    assert(y.value == 0.)

def test_sub_np_ad():
    x_np = np.arange(4, dtype=np.float)
    x_ad = ADTypeT(1.)

    y = x_np - x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == x_np[i] - 1.)
    
    y = x_ad - x_np
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == 1. - x_np[i])

def test_sub_npad_bs():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i))
    x_bs = 1.

    y = x_np_ad - x_bs
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) - 1.)

    y = x_bs - x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == 1. - float(i))

def test_sub_npad_ad():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i))
    x_ad = ADTypeT(1.)

    y = x_np_ad - x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) - 1.)

    y = x_ad - x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == 1. - float(i))

def test_sub_npad_np():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i))
    x_np = np.arange(4, dtype=np.float)

    y = x_np_ad - x_np
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) - float(i))

    y = x_np - x_np_ad
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) - float(i))

def test_sub_npad_npad():
    x_np_ad1 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad1[i] = ADTypeT(float(i))
    x_np_ad2 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad2[i] = ADTypeT(float(i))

    y = x_np_ad1 - x_np_ad2
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) - float(i))

"""
MUL TESTS
---------
"""
def test_mul_bs_ad():
    x_bs = 2.
    x_ad = ADTypeT(3.)

    y = x_bs * x_ad
    assert(type(y) is ADTypeT)
    assert(y.value == 6.)
    
    y = x_ad * x_bs
    assert(type(y) is ADTypeT)
    assert(y.value == 6.)

def test_mul_ad_ad():
    x_ad1 = ADTypeT(2.)
    x_ad2 = ADTypeT(3.)

    y = x_ad1 * x_ad2
    assert(type(y) is ADTypeT)
    assert(y.value == 6.)

def test_mul_np_ad():
    x_np = np.arange(4, dtype=np.float)
    x_ad = ADTypeT(2.)

    y = x_np * x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == x_np[i]*2.)
    
    y = x_ad * x_np
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == x_np[i]*2.)

def test_mul_npad_bs():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i))
    x_bs = 2.

    y = x_np_ad * x_bs
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) * 2.)

    y = x_bs * x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) * 2.)

def test_mul_npad_ad():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i))
    x_ad = ADTypeT(2.)

    y = x_np_ad * x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) * 2.)

    y = x_ad * x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) * 2.)

def test_mul_npad_np():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i))
    x_np = np.arange(4, dtype=np.float)

    y = x_np_ad * x_np
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) * float(i))

    y = x_np * x_np_ad
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) * float(i))

def test_mul_npad_npad():
    x_np_ad1 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad1[i] = ADTypeT(float(i))
    x_np_ad2 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad2[i] = ADTypeT(float(i))

    y = x_np_ad1 * x_np_ad2
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i) * float(i))

# truediv tests
"""
TRUEDIV TESTS
-------------
"""
def test_truediv_bs_ad():
    x_bs = 2.
    x_ad = ADTypeT(3.)

    y = x_bs / x_ad
    assert(type(y) is ADTypeT)
    assert(y.value == 2./3.)
    
    y = x_ad / x_bs
    assert(type(y) is ADTypeT)
    assert(y.value == 3./2.)

def test_truediv_ad_ad():
    x_ad1 = ADTypeT(2.)
    x_ad2 = ADTypeT(3.)

    y = x_ad1 / x_ad2
    assert(type(y) is ADTypeT)
    assert(y.value == 2./3.)

def test_truediv_np_ad():
    x_np = np.arange(4, dtype=np.float) + 1.
    x_ad = ADTypeT(2.)

    y = x_np / x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == x_np[i]/2.)
    
    y = x_ad / x_np
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == 2./x_np[i])
        
def test_truediv_npad_bs():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i + 1))
    x_bs = 2.

    y = x_np_ad / x_bs
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 1) / 2.)

    y = x_bs / x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == 2. / float(i + 1))

def test_truediv_npad_ad():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i + 1))
    x_ad = ADTypeT(2.)

    y = x_np_ad / x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 1) / 2.)

    y = x_ad / x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == 2./ float(i + 1))

def test_truediv_npad_np():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i + 1))
    x_np = np.arange(4, dtype=np.float) + 2

    y = x_np_ad / x_np
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 1) / float(i + 2))

    y = x_np / x_np_ad
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 2) / float(i + 1))

def test_truediv_npad_npad():
    x_np_ad1 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad1[i] = ADTypeT(float(i + 1))
    x_np_ad2 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad2[i] = ADTypeT(float(i + 2))

    y = x_np_ad1 / x_np_ad2
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 1) / float(i + 2))

"""
POW TESTS
---------
"""
def test_pow_bs_ad():
    x_bs = 2.
    x_ad = ADTypeT(3.)

    y = x_bs ** x_ad
    assert(type(y) is ADTypeT)
    assert(y.value == 2.**3.)
    
    y = x_ad ** x_bs
    assert(type(y) is ADTypeT)
    assert(y.value == 3.**2.)

def test_pow_ad_ad():
    x_ad1 = ADTypeT(2.)
    x_ad2 = ADTypeT(3.)

    y = x_ad1 ** x_ad2
    assert(type(y) is ADTypeT)
    assert(y.value == 2.**3.)

def test_truediv_np_ad():
    x_np = np.arange(4, dtype=np.float) + 1.
    x_ad = ADTypeT(2.)

    y = x_np ** x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == x_np[i]**2.)
    
    y = x_ad ** x_np
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == 2.**x_np[i])
        
def test_pow_npad_bs():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i + 1))
    x_bs = 2.

    y = x_np_ad ** x_bs
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 1) ** 2.)

    y = x_bs ** x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == 2. ** float(i + 1))

def test_pow_npad_ad():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i + 1))
    x_ad = ADTypeT(2.)

    y = x_np_ad ** x_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 1) ** 2.)

    y = x_ad ** x_np_ad
    for i in range(0, 4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == 2. ** float(i + 1))

def test_pow_npad_np():
    x_np_ad = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad[i] = ADTypeT(float(i + 1))
    x_np = np.arange(4, dtype=np.float) + 2

    y = x_np_ad ** x_np
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 1) ** float(i + 2))

    y = x_np ** x_np_ad
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 2) ** float(i + 1))

def test_pow_npad_npad():
    x_np_ad1 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad1[i] = ADTypeT(float(i + 1))
    x_np_ad2 = np.empty(4, dtype=ADTypeT)
    for i in range(4):
        x_np_ad2[i] = ADTypeT(float(i + 2))

    y = x_np_ad1 ** x_np_ad2
    for i in range(4):
        assert(type(y[i]) is ADTypeT)
        assert(y[i].value == float(i + 1) ** float(i + 2))