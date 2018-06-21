#!/usr/bin/env python
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse import BCOO
from sparse.utils import assert_eq
from sparse.bcoo import bcalc

import pytest


@pytest.mark.parametrize('shape_x, block_shape_x, shape_y, block_shape_y, descr', [
    [(8,6), (2,2), (8,4,6), (2,2,2), "ij,ikj->k"],
    [(8,4,9), (1,2,3), (9,4,6), (3,2,2), "ijk,kmn->ijmn"],
])
def test_normal_einsum(shape_x, block_shape_x, shape_y, block_shape_y, descr):
    x = sparse.brandom(shape_x, block_shape_x, 0.3, format='bcoo')
    x_d = x.todense()

    y = sparse.brandom(shape_y, block_shape_y, 0.3, format='bcoo')
    y_d = y.todense()

    c= bcalc.einsum(descr, x, y, DEBUG=False)
    elemC = np.einsum(descr, x_d, y_d)
    assert_eq(elemC, c)


def test_einsum_with_transpose():
    shape_x = (8,6)
    block_shape_x = (2,2)
    x = sparse.brandom(shape_x, block_shape_x, 0.3, format='bcoo').T
    x_d = x.todense()

    shape_y = (8,4,6)
    block_shape_y = (2,2,2)
    y = sparse.brandom(shape_y, block_shape_y, 0.3, format='bcoo').transpose((2,0,1))
    y_d = y.todense()

    c= bcalc.einsum("ij,ijk->k", x, y, DEBUG=False)
    elemC = np.einsum("ij,ijk->k", x_d, y_d)
    assert_eq(elemC, c)


    shape_x = (8,4,9)
    block_shape_x = (1,2,3)
    x = sparse.brandom(shape_x, block_shape_x, 0.2, format='bcoo').transpose((1,0,2))
    x_d = x.todense()

    shape_y = (9,4,6)
    block_shape_y = (3,2,2)
    y = sparse.brandom(shape_y, block_shape_y, 0.5, format='bcoo').T
    y_d = y.todense()

    c= bcalc.einsum("jik,nmk->minj", x, y, DEBUG=False)
    elemC = np.einsum("jik,nmk->minj", x_d, y_d)
    assert_eq(elemC, c)

    c= bcalc.einsum("jik,nmk->ijmn", x, y, DEBUG=False)
    elemC = np.einsum("jik,nmk->ijmn", x_d, y_d)
    assert_eq(elemC, c)


def test_einsum_as_transpose():
    # swap axes
    shape_y = (8,4,6)
    block_shape_y = (2,2,2)
    y = sparse.brandom(shape_y, block_shape_y, 0.3, format='bcoo').transpose((2,0,1))
    y_d = y.todense()

    c= bcalc.einsum("ijk->kji", y)
    elemC = np.einsum("ijk->kji", y_d)
    assert_eq(elemC, c)

    c= bcalc.einsum("ijk->ijk", y)
    elemC = np.einsum("ijk->ijk", y_d)
    assert_eq(elemC, c)


#TODO:def test_einsum_views():
#TODO:    shape_y = (6,6,6)
#TODO:    block_shape_y = (2,2,2)
#TODO:    y = sparse.brandom(shape_y, block_shape_y, 0.3, format='bcoo').transpose((0,2,1))
#TODO:    y_d = y.todense()
#TODO:
#TODO:    c= bcalc.einsum("ijj->ij", y)
#TODO:    elemC = np.einsum("ijj->ij", y_d)
#TODO:    assert_eq(elemC, c)
#TODO:
#TODO:    c= bcalc.einsum("iii->i", y)
#TODO:    elemC = np.einsum("iii->i", y_d)
#TODO:    assert_eq(elemC, c)

def test_einsum_misc():
    shape_x = (4,2,2)
    block_shape_x = (2,2,2)
    x = sparse.brandom(shape_x, block_shape_x, 0.5, format='bcoo')
    x_d = x.todense()

    shape_y = (2,2)
    block_shape_y = (2,1)
    y = sparse.brandom(shape_y, block_shape_y, 0.5, format='bcoo')
    y_d = y.todense()

    shape_z = (2,2)
    block_shape_z = (1,2)
    z = sparse.brandom(shape_z, block_shape_z, 0.5, format='bcoo')
    z_d = z.todense()

    c = bcalc.einsum('kxy,yz->kxz', x, y)
    elemC = np.einsum('kxy,yz->kxz', x_d, y_d)
    assert_eq(elemC, c)

    c = bcalc.einsum('kxy,yz,zx->k', x, y, z)
    elemC = np.einsum('kxy,yz,zx->k', x_d, y_d, z_d)
    assert_eq(elemC, c)

def test_einsum_shape_error():
    # test for shape error while block_shape is ok
    shape_x = (4,2,2)
    block_shape_x = (2,2,2)
    x = sparse.brandom(shape_x, block_shape_x, 0.5, format='bcoo')
    x_d = x.todense()

    shape_y = (1,1)
    block_shape_y = (1,1)
    y = sparse.brandom(shape_y, block_shape_y, 0.5, format='bcoo')
    y_d = y.todense()

    shape_z = (2,1)
    block_shape_z = (2,1)
    z = sparse.brandom(shape_z, block_shape_z, 0.5, format='bcoo')
    z_d = z.todense()

    with pytest.raises(RuntimeError):
        c = bcalc.einsum('kxy,yz,zx->k', x, y, z)


    # test for block_shape error while outer_shape is ok
    shape_x = (4,4,2)
    block_shape_x = (2,2,2)
    x = sparse.brandom(shape_x, block_shape_x, 0.5, format='bcoo')
    x_d = x.todense()

    shape_y = (1,1)
    block_shape_y = (1,1)
    y = sparse.brandom(shape_y, block_shape_y, 0.5, format='bcoo')
    y_d = y.todense()

    shape_z = (2,2)
    block_shape_z = (2,1)
    z = sparse.brandom(shape_z, block_shape_z, 0.5, format='bcoo')
    z_d = z.todense()

    with pytest.raises(RuntimeError):
        c = bcalc.einsum('kxy,yz,zx->k', x, y, z)


def test_einsum_zeros():
    shape_x = (5,1,4,2,3)
    block_shape_x = (1,1,2,2,3)
    x = sparse.bcoo.zeros(shape_x, np.complex, block_shape_x)
    x_d = x.todense()

    shape_y = (5,1,11)
    block_shape_y = (1,1,1)
    y = sparse.bcoo.zeros(shape_y, np.float32, block_shape_y)
    y_d = y.todense()

    c= bcalc.einsum('ijklm,ijn->lnmk', x, y)
    elemC = np.einsum('ijklm,ijn->lnmk', x_d, y_d)
    assert_eq(elemC, c)


def test_einsum_mix_types():
    shape_x = (5,1,4,2,3)
    block_shape_x = (1,1,2,2,3)
    x = sparse.brandom(shape_x, block_shape_x, 0.5)
    x.data = x.data + 1j
    x_d = x.todense()

    shape_y = (5,1,11)
    block_shape_y = (1,1,1)
    y = sparse.brandom(shape_y, block_shape_y, 0.5)
    y.data = y.data.astype(np.float32)
    y_d = y.todense()

    c= bcalc.einsum('ijklm,ijn->lnmk', x, y)
    elemC = np.einsum('ijklm,ijn->lnmk', x_d, y_d)
    assert_eq(elemC, c)

    c= bcalc.einsum('ijklm,ijn->lnmk', x_d, y)
    elemC = np.einsum('ijklm,ijn->lnmk', x_d, y_d)
    assert_eq(elemC, c)

if __name__ == '__main__':
    test_einsum_zeros()
    test_einsum_mix_types()

