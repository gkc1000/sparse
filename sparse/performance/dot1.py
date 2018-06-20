#!/usr/bin/env python
import time
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse import BCOO
from sparse.bcoo import bcalc

print(time.clock())

def dot1(shape_x, block_shape_x, shape_y, block_shape_y):
    x = sparse.brandom(shape_x, block_shape_x, .3, format='bcoo')
    y = sparse.brandom(shape_y, block_shape_y, .3, format='bcoo')
    c = bcalc.einsum("ij,pjk->pik", x, y, DEBUG=False)


shape_x = (2000,2000)
block_shape_x = (8,8)
shape_y = (40,2000,40)
block_shape_y = (8,8,8)

print(time.clock())
dot1(shape_x, block_shape_x, shape_y, block_shape_y)
print(time.clock())
exit()

from pyscf import lib
a1 = np.ones(shape_x)
a2 = np.ones(shape_y)
print(time.clock())
lib.einsum('ij,pjk->ipk', a1, a2)
print(time.clock())
