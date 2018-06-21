#!/usr/bin/env python
import numpy as np

import sparse
from sparse import BCOO
from sparse.utils import assert_eq
from sparse.bcoo import bcalc

#import pytest
import time


shape_x, block_shape_x, shape_y, block_shape_y, descr = [(1000,1000), (50,50), (1000,1000,1000), (50,50,50), "ij,ijk->k"] 

print "generate x"
x = sparse.brandom(shape_x, block_shape_x, 0.3, format='bcoo')
x_d = x.todense()
#np.save('./x_d.npy',x_d)

print "generate y"
y = sparse.brandom(shape_y, block_shape_y, 0.3, format='bcoo')
y_d = y.todense()
#np.save('./y_d.npy',y_d)
print "tensors are generated"

time1 = time.time()
c= bcalc.einsum(descr, x, y, DEBUG=False)
time2 = time.time()

print "block einsum time:"
print time2 - time1

time3 = time.time()
elemC = np.einsum(descr, x_d, y_d)
time4 = time.time()

print "numpy einsum time:"
print time4 - time3



