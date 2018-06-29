#!/usr/bin/env python
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse import BCOO
from sparse.utils import assert_eq

#import pytest

def test_add_simple():
    data = np.arange(1,4).repeat(4).reshape(3,2,2)
    coords = np.array([[1,1,0],[0,1,1]])
    x = BCOO(coords, data = data, shape = (4,4), block_shape = (2,2))
    x_d = x.todense()
    y = x + x
    y_d = x_d + x_d
    assert_eq(y, y_d)

def test_add_multi_dim_array():
    x = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    x_d = x.todense()
    y = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    y_d = y.todense()
    
    z = x + y
    z_d = x_d + y_d
    assert_eq(z, z_d)


if __name__ == '__main__':
    print("\n main test \n")
    test_add_simple()
    test_add_multi_dim_array()
