#!/usr/bin/env python
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse import BCOO
from sparse.utils import assert_eq

import pytest

def test_brandom():
    x = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    y = x.todense()
    assert_eq(x, y)

def test_from_numpy():
    #a = np.random.random((6,5,4,1))
    a = np.zeros((6,5,4,1))
    x = BCOO.from_numpy(a, block_shape = (2,5,2,1))
    assert_eq(a,x)

def test_zero_size():
    x = sparse.brandom((0,0,0), (2,2,2))
    assert(x.nnz == 0)
    x = sparse.bcoo.zeros((0,0,0), block_shape=(2,2,2))
    assert(x.nnz == 0)


@pytest.mark.parametrize('shape, dtype, block_shape', [
    [(4,2,4), np.int32, (1,2,2)],
    [(4,4), np.complex128, (1,2)],
    [(4,4), np.float32, (1,2)],
    [(4,4), np.dtype([('a', np.int), ('b', np.float)]), (1,2)],
    [(4, 9, 16), np.dtype('i4,(3,2)f'), (2, 3, 4)],
])
def test_zeros(shape, dtype, block_shape):
    x = sparse.bcoo.zeros(shape, dtype, block_shape)
    assert(x.shape == shape)
    assert(x.block_shape == block_shape)
    assert(x.dtype == dtype)
    assert(x.nnz == 0)
    assert(x.block_nnz == 0)


def test_invalid_shape_error():
    with pytest.raises(RuntimeError):
        sparse.brandom((3, 4), block_shape=(2, 3), format='bcoo')


@pytest.mark.parametrize('axis', [
    None,
    (1, 2, 0),
    (2, 1, 0),
    (0, 1, 2),
    (0, 1, -1),
    (0, -2, -1),
    (-3, -2, -1),
])
def test_transpose(axis):
    x = sparse.brandom((6, 2, 4), (2,2,2), density=0.3)
    y = x.todense()
    xx = x.transpose(axis)
    yy = y.transpose(axis)
    assert_eq(xx, yy)


def test_block_reshape():

    a = np.array([[1, -1, 0, 0], [1 , -1 , 0, 0], [2,3 ,6,7], [4,5,8,9]])
    x = BCOO.from_numpy(a, block_shape = (2,2))
    y = x.todense()

    outer_shape_new = (1,4)
    block_shape_new = (2,2) # unchanged
    z = x.block_reshape(outer_shape_new, block_shape_new)

    print("original matrix (2,2)")
    print(y)
    print("block reshaped matrix (1,4)")
    print(z.todense())


@pytest.mark.parametrize('a, a_bshape, axis, b, b_bshape', [
    #FIXME[(4, 6)      , (2, 3)      , (0, 1)         , (24,)   , (3,)   ],
    [(6, 8)      , (3, 4)      , (0, 1)         , (6, 8)  , (3, 4 )],
    [(6, 8)      , (3, 4)      , (1, 0)         , (8, 6)  , (4, 3 )],
    [(6, 8)      , (3, 4)      , (0, 1)         , (-1, 8) , (3, 4 )],
    [(6, 8)      , (3, 4)      , (1, 0)         , (8, -1) , (4, 3 )],
    #FIXME[(6, 6, 4)   , (2, 3, 4)   , (0, -2, -1)    , (6, 24) , (2, 12)],
    #FIXME[(6, 6, 4)   , (2, 3, 4)   , (0, 2, 1)      , (-1, 6) , (8, 3 )],
    #FIXME[(6, 6, 4)   , (2, 3, 4)   , (2, 1, 0)      , (24, 6) , (12, 2)],
    #FIXME[(6, 6, 4, 5), (2, 3, 4, 5), (0, -2, 3, -3) , (180, 4), (30, 4)],
    #FIXME[(6, 6, 4, 5), (2, 3, 4, 5), (1, 3, 0, 2)   , (30, -1), (15, 8)],
    #FIXME[(6, 6, 4, 5), (2, 3, 4, 5), (2, 1,-4, 3)   , (-1, 5) , (24, 5)],
])
def test_transpose_reshape(a, a_bshape, axis, b, b_bshape):
    x = sparse.brandom(a, a_bshape, density=0.3)
    y = x.todense()
    xx = x.transpose(axis).reshape(b, b_bshape)
    yy = y.transpose(axis).reshape(b)
    assert_eq(xx, yy)


def test_todense():
    s = sparse.bcoo.zeros((4, 9, 16), 'D', block_shape=(2, 3, 4))
    s.todense()

    s = sparse.bcoo.zeros((), block_shape=())
    s.todense()

    s = sparse.bcoo.zeros((4, 9, 16), 'D', block_shape=(2, 3, 4))
    x = s.getblock((1,1,1,Ellipsis))
    x.todense()


def test_tobsr():
    data = np.arange(1,7).repeat(4).reshape((-1,2,2))
    coords = np.array([[0,0,0,2,1,2],[0,1,1,0,2,2]])
    block_shape = (2,2)
    shape = (8,6)
    x = BCOO(coords, data=data, shape=shape, block_shape=block_shape) 
    y = x.todense()
    z = x.tobsr()
    assert_eq(z, y)


if __name__ == '__main__':
    print("\n main test \n")
    test_brandom()
    test_from_numpy()
    test_transpose(None)
    test_block_reshape()
    test_tobsr()
