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
    x = np.random.rand(2, 4, 4)
    s = BDOK(x, (2, 2, 2))

    #x = np.array([[1,-1,0,0],[1,-1,0,0],[2,2,3,3],[2,2,3,3]])
    #s = BDOK(x, (2, 2))


    assert_eq(x, s)


def test_convert_to_numpy():
    s = sparse.brandom((4, 6, 8), (2, 3, 4), 0.5, format='bdok')
    x = s.todense()

    assert_eq(x, s)

def test_getitem():
    s = sparse.brandom((4, 6, 8), (2, 3, 4), 0.5, format='bdok')
    x = s.todense()
    x_b = s.to_bumpy()
    shape = (2,2,2)


    #x = np.array([[1,-1,0,0],[1,-1,0,0],[2,2,3,3],[2,2,3,3]])
    #s = BDOK(x, (2, 2))
    #x_b = s.to_bumpy()
    #shape = (2,2)


    for _ in range(s.nnz):
        idx = np.random.randint(np.prod(shape))
        idx = np.unravel_index(idx, shape)

        assert np.allclose(s[idx], x_b[idx])

def test_setitem():
    s = sparse.brandom((6, 6, 8), (2, 3, 4), 0.5, format='bdok')
    x = s.todense()
    x_b = s.to_bumpy()

    shape = (3,2,2)
    value = np.random.random((2,3,4))
    idx = np.random.randint(np.prod(shape))
    idx = np.unravel_index(idx, shape)

    s[idx] = value
    x_b[idx] = value

    assert_eq(x_b.todense(), s.todense())

def test_default_dtype():
    s = BDOK((5,), block_shape = (1,))

    assert s.dtype == np.float64


def test_int_dtype():
    data = {
        1: np.uint8(1),
        2: np.uint16(2),
    }

    s = DOK((5,), data)

    assert s.dtype == np.uint16


def test_float_dtype():
    data = {
        1: np.uint8(1),
        2: np.float32(2),
    }

    s = BDOK((5,), block_shape = (1,), data = data)

    assert s.dtype == np.float32


def test_set_zero():
    s = BDOK((1,), block_shape = (1,), dtype=np.uint8)
    s[0] = 1
    s[0] = 0

    assert s[0] == 0
    assert s.nnz == 0


def test_asformat():
    s = sparse.brandom((4, 6, 8), (2, 3, 4), 0.5, format='bdok')
    s2 = s.asformat('bcoo')
    #s2 = s.asformat('bdok')

    #s = sparse.brandom((4, 3), (2, 1), 0.5, format='bdok')
    #x = np.array([[1,-1,0,0],[1,-1,0,0],[2,2,3,3],[2,2,3,3]])
    #s = BDOK(x, block_shape = (2, 2))
    #s2 = s.asformat('bcoo')

    assert_eq(s, s2)

if __name__ == '__main__':
    print("\n main test \n")
    test_convert_to_numpy()
    test_convert_from_numpy()
    test_getitem()
    test_setitem()
    test_default_dtype()
    test_float_dtype()
    test_set_zero()
    test_asformat()
