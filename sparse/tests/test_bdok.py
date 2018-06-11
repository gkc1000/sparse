#!/usr/bin/env python
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse.utils import assert_eq


def test_random_shape_nnz():
    shape = (4,2)
    block_shape = (2,2)
    density = 0.1 

    s = sparse.brandom(shape, block_shape, density, format='bdok')

    assert isinstance(s, BDOK)

    assert s.shape == shape
    expected_nnz = density * np.prod(shape)
    assert np.floor(expected_nnz) <= s.nnz <= np.ceil(expected_nnz)

    
def test_convert_to_coo():
    s1 = sparse.brandom((4, 6, 8), (2, 3, 4), 0.5, format='bdok')
    s2 = sparse.BCOO(s1)

    assert_eq(s1, s2)

    
def test_convert_from_coo():
    s1 = sparse.brandom((6, 6, 12), (2, 2, 3), 0.5, format='bcoo')
    s2 = BDOK(s1)

    assert_eq(s1, s2)


def test_convert_from_numpy():
    #x = np.random.rand(2, 4, 4)
    #s = BDOK(x, (2, 2, 2))
    
    x = np.array([[1,-1,0,0],[1,-1,0,0],[2,2,3,3],[2,2,3,3]])
    s = BDOK(x, (2, 2))

    print x
    print s.todense()

    assert_eq(x, s)


def test_convert_to_numpy():
    s = sparse.brandom((4, 6, 8), (2, 3, 4), 0.5, format='bdok')
    x = s.todense()

    assert_eq(x, s)


    
    
if __name__ == '__main__':
    print "\n main test \n"
    test_convert_from_numpy()
