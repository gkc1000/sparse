#!/usr/bin/env python
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse import BCOO
from sparse.utils import assert_eq

#import pytest

def test_add():
    #x = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    #y = x.todense()
    data = np.arange(1,4).repeat(4).reshape(3,2,2)
    coords = np.array([[1,1,0],[0,1,1]])
    x = BCOO(coords, data = data, shape = (4,4), block_shape = (2,2))
    #print x
    #print x.todense()
    #'''
    y = x + x
    print y
    print y.todense()
    #'''

if __name__ == '__main__':
    print("\n main test \n")
    test_add()
