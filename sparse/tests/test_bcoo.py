#!/usr/bin/env python
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse import BCOO
from sparse.utils import assert_eq

def test_brandom():
    x = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    y = x.todense()
    assert_eq(x, y)
    



def test_transpose(axes):
    #x = sparse.brandom((2, 3, 4),  ,density=.25)
    #x = sparse.brandom((4, 6, 8), (2, 3, 4), 0.5, format='bcoo')
    #x = sparse.brandom((4, 4), (2, 2), 1.0, format='bcoo')
    a = np.array([[1, -1, 0, 0], [1 , -1 , 0, 0], [2,3 ,6,7], [4,5,8,9]])
    x = BCOO.from_numpy(a, block_shape = (2,2))
    y = x.todense()
  
    xx = x.transpose(axes)
    yy = y.transpose(axes)

    assert_eq(xx, yy)

    # ZHC TODO: the brandom for bcoo format seems not correct.


def test_block_reshape():
    
    a = np.array([[1, -1, 0, 0], [1 , -1 , 0, 0], [2,3 ,6,7], [4,5,8,9]])
    x = BCOO.from_numpy(a, block_shape = (2,2))
    y = x.todense()
    
    outer_shape_new = (1,4)
    block_shape_new = (2,2) # unchanged
    z = x.block_reshape(outer_shape_new, block_shape_new)

    print "original matrix (2,2)"
    print y
    print "block reshaped matrix (1,4)"
    print z.todense()

    
if __name__ == '__main__':
    print "\n main test \n"
    test_brandom()
    test_transpose(None)
    test_block_reshape()
