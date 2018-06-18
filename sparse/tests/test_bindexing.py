#!/usr/bin/env python
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse import BCOO
from sparse.utils import assert_eq

import pytest


@pytest.mark.parametrize('index, npindex', [
    # Integer
    [0, slice(0,2)],
    [1, slice(2,4)],
    [-1, slice(-2,None)],
    [(1, 1, 1), (slice(2,4), slice(3,6), slice(4,8))],
    # Pure slices
    [(slice(0, 2),), slice(0, 4)],
    [(slice(None, 2), slice(None, 2)), (slice(0,4), slice(0,6))],
    [(slice(1, None), slice(1, None)), (slice(2,None), slice(3,None))],
    [(slice(None, None),), slice(None,None)],
    [(slice(None, None, -1),), lambda x:x.reshape(2,2,9,16)[::-1].reshape(-1,9,16)],
    [(slice(None, 2, -1), slice(None, 2, -1)), lambda x:x.reshape(2,2,3,3,16)[:2:-1,:,:2:-1].reshape(0,0,16)],
    [(slice(1, None, 2), slice(1, None, 2)), lambda x:x.reshape(2,2,3,3,16)[1::-2,:,1::-2].reshape(2,-1,16)],
    [(slice(None, None, 2),), lambda x:x.reshape(2,2,9,16)[::2].reshape(-1,9,16)],
    [(slice(None, 2, -1), slice(None, 2, -2)), lambda x:x.reshape(2,2,3,3,16)[:2:-1,:,:2:-2].reshape(0,0,16)],
    [(slice(1, None, 2), slice(1, None, 1)), lambda x:x.reshape(2,2,3,3,16)[1::2,:,1::1].reshape(-1,6,16)],
    [(slice(None, None, -2),), lambda x:x.reshape(2,2,9,16)[::-2].reshape(-1,9,16)],
    # Combinations
    [(0, slice(0, 2),), (slice(0,2), slice(0,6))],
    [(slice(0, 1), 0), (slice(0,2), slice(0,3))],
    [(None, slice(1, 3), 0), (None, slice(2,6), slice(0,3))],
    [(slice(0, 3), None, 0), (slice(0,6), None, slice(0,3))],
    [(slice(1, 2), slice(2, 4)), (slice(2,4), slice(6,12))],
    [(slice(1, 2), slice(None, None)), (slice(2,4), slice(None,None))],
    [(slice(1, 2), slice(None, None), 2), (slice(2,4), slice(None,None), slice(8,12))],
    [(slice(1, 2, 2), slice(None, None), 2), lambda x:x.reshape(2,2,9,16)[1::2,:,:,8:12].reshape(-1,9,4)],
    [(slice(1, 2, None), slice(None, None, 2), 2), lambda x:x.reshape(4,3,3,16)[2:4,::2,:,8:12].reshape(2,-1,4)],
    [(slice(1, 2, -2), slice(None, None), -2), lambda x:x.reshape(2,2,9,16)[1:2:-2,:,:,-8:-4].reshape(-1,9,4)],
    [(slice(1, 2, None), slice(None, None, -2), 2), lambda x:x.reshape(4,3,3,16)[2:4,::-2,:,8:12].reshape(2,-1,4)],
    [(slice(1, 2, -1), slice(None, None), -1), lambda x:x.reshape(2,2,9,16)[1:2:-1,:,:,-4:].reshape(-1,9,4)],
    [(slice(1, 2, None), slice(None, None, -1), 2), lambda x:x.reshape(4,3,3,16)[2:4,::-1,:,8:12].reshape(2,-1,4)],
    [(slice(2, 0, -1), slice(None, None), -1), lambda x:x.reshape(2,2,9,16)[2:0:-1,:,:,-4:].reshape(-1,9,4)],
    [(slice(-2, None, None),), slice(-4,None)],
    [(slice(-1, None, None), slice(-2, None, None)), (slice(-2,None), slice(-6,None))],
    # With ellipsis
    [(Ellipsis, slice(1, 3)), (Ellipsis, slice(4,12))],
    [(1, Ellipsis, slice(1, 3)), (slice(2,4), Ellipsis, slice(4,12))],
    [(slice(0, 1), Ellipsis), (slice(0,2), Ellipsis)],
    [(Ellipsis, None), (Ellipsis, None)],
    [(None, Ellipsis), (None, Ellipsis)],
    [(1, Ellipsis), (slice(2,4), Ellipsis)],
    [(1, Ellipsis, None), (slice(2,4), Ellipsis, None)],
    [(1, 1, 1, Ellipsis), (slice(2,4), slice(3,6), slice(4,8), Ellipsis)],
    [(Ellipsis, 1, None), (Ellipsis, slice(4,8), None)],
    # Pathological - Slices larger than array
    [(slice(None, 1000)), (slice(None,2000))],
    [(slice(None), slice(None, 1000)), (slice(None), slice(None,3000))],
    [(slice(None), slice(1000, -1000, -1)), lambda x:x.reshape(4,3,3,16)[:,1000:-1000:-1].reshape(4,-1,16)],
    [(slice(None), slice(1000, -1000, -50)), lambda x:x.reshape(4,3,3,16)[:,1000:-1000:-50].reshape(4,-1,16)],
    # Pathological - Wrong ordering of start/stop
    [(slice(5, 0),), (slice(10,0),)],
    [(slice(0, 5, -1),), lambda x:x.reshape(2,2,9,16)[0:5:-1].reshape(-1,9,16)],
])
def test_block_slicing(index, npindex):
    s = sparse.brandom((4, 9, 16), (2, 3, 4), density=0.5)
    x = s.todense()
    if callable(npindex):
        assert_eq(s.getblock(index), npindex(x))
    else:
        assert_eq(s.getblock(index), x[npindex])

def test_custom_dtype_block_slicing():
    dt = np.dtype([('part1', np.float_),
                   ('part2', np.int_, (2,)),
                   ('part3', np.int_, (2, 2))])

    s = sparse.bcoo.zeros((4, 9, 16), dt, block_shape=(2, 3, 4))
    assert s.getblock('part1').shape == (4,9,16)
    assert s.getblock('part2').shape == (4,9,16,2)
    assert s.getblock('part3').shape == (4,9,16,2,2)


