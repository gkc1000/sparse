#!/usr/bin/env python
import numpy as np

import sparse
from sparse import BCOO
from sparse.utils import assert_eq
from sparse.bcoo import bcalc

#import pytest
import time
from pyscf.lib import numpy_helper as nh

shape_x, block_shape_x, shape_y, block_shape_y, descr = [(400,400,400), (20,20,20), (400,400,400), (20,20,20), "ijl,ikj->kl"] 

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
#c= bcalc.einsum_old(descr, x, y, DEBUG=False)
c= bcalc.einsum(descr, x, y, DEBUG=False)
time2 = time.time()

print "block einsum time:"
print time2 - time1

#time3 = time.time()
#elemC = np.einsum(descr, x_d, y_d)
#time4 = time.time()

#print "numpy einsum time:"
#print time4 - time3

time5 = time.time()
elemC2 = nh.einsum(descr, x_d, y_d)
time6 = time.time()

print "numpy pyscf einsum time:"
print time6 - time5


