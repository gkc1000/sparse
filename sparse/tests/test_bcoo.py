#!/usr/bin/env python
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse import BCOO
from sparse.utils import assert_eq


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
    # transpose for axes which break the block structure is not considered currently/
    # the axes here have length len(block_shape)


    
if __name__ == '__main__':
    print "\n main test \n"
    test_transpose(None)
