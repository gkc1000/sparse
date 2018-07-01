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

def test_add_mismatch_block():
    data_x = np.arange(1,4).repeat(4).reshape(3,2,2)
    coords_x = np.array([[1,1,0],[0,1,1]])
    x = BCOO(coords_x, data = data_x, shape = (4,4), block_shape = (2,2))
    x_d = x.todense()
    data_y = np.array([[[-2.0, -2.5], [-2,-2]],[[-2,-2],[-2,-2]]])
    coords_y = np.array([[0,1],[0,1]])
    y = BCOO(coords_y, data = data_y, shape = (4,4), block_shape = (2,2))
    y_d = y.todense()
    z = x + y
    z_d = x_d + y_d
    assert_eq(z, z_d)
   
def test_add_zero():
    # the result is zero
    data_x = np.arange(3,6).repeat(4).reshape(3,2,2).astype(np.double)
    coords_x = np.array([[1,1,0],[0,1,1]])
    x = BCOO(coords_x, data = data_x, shape = (4,4), block_shape = (2,2))
    x_d = x.todense()
    y = -x
    y_d = y.todense()
    z = x + y
    z_d = x_d + y_d
    assert_eq(z, z_d)

    # all add numbers are zero
    a = np.zeros((6,5,4,1), dtype = np.complex)
    x = BCOO.from_numpy(a, block_shape = (2,5,2,1))
    x_d = x.todense()
    y = BCOO.from_numpy(a, block_shape = (2,5,2,1))
    y_d = y.todense()
    z = x + y
    z_d = x_d + y_d
    assert_eq(z, z_d)

def test_scaling():
    data_x = np.arange(3,6).repeat(4).reshape(3,2,2).astype(np.double)
    coords_x = np.array([[1,1,0],[0,1,1]])
    x = BCOO(coords_x, data = data_x, shape = (4,4), block_shape = (2,2))
    x_d = x.todense()
    z = x * -3.1
    z_d = x_d * -3.1
    assert_eq(z, z_d)
    
    x = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    x_d = x.todense()
    scaling_factor = np.random.random()
    z = x * scaling_factor 
    z_d = x_d * scaling_factor
    assert_eq(z, z_d)

    x = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    x_d = x.todense()
    scaling_factor = 0
    z = x * scaling_factor 
    z_d = x_d * scaling_factor
    assert_eq(z, z_d)

def test_subtraction():
    x = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    x_d = x.todense()
    y = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    y_d = y.todense()
    z = x - y
    z_d = x_d - y_d
    assert_eq(z, z_d)
    
    data = np.arange(1,4).repeat(4).reshape(3,2,2)
    coords = np.array([[1,1,0],[0,1,1]])
    x = BCOO(coords, data = data, shape = (4,4), block_shape = (2,2))
    x_d = x.todense()
    y = x - x
    y_d = x_d - x_d
    assert_eq(y, y_d)


def test_multiply():
    x = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    x_d = x.todense()
    y = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    y_d = y.todense()
    z = x * y
    z_d = x_d * y_d
    assert_eq(z, z_d)
    
    data = np.arange(1,4).repeat(4).reshape(3,2,2)
    coords = np.array([[1,1,0],[0,1,1]])
    x = BCOO(coords, data = data, shape = (4,4), block_shape = (2,2))
    x_d = x.todense()
    y = x * x
    y_d = x_d * x_d
    assert_eq(y, y_d)
    
    data_x = np.arange(1,4).repeat(4).reshape(3,2,2)
    coords_x = np.array([[1,1,0],[0,1,1]])
    x = BCOO(coords_x, data = data_x, shape = (4,4), block_shape = (2,2))
    x_d = x.todense()
    data_y = np.array([[[-2.0, -2.5], [-2,-2]],[[-2,-2],[-2,-2]]])
    coords_y = np.array([[0,1],[0,1]])
    y = BCOO(coords_y, data = data_y, shape = (4,4), block_shape = (2,2))
    y_d = y.todense()
    z = x * y
    z_d = x_d * y_d
    assert_eq(z, z_d)



if __name__ == '__main__':
    print("\n main test \n")
    test_add_simple()
    test_add_multi_dim_array()
    test_add_mismatch_block()
    test_add_zero()
    test_scaling()
    test_subtraction()
