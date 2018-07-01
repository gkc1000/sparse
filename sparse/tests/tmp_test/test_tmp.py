#!/usr/bin/env python
import numpy as np

import sparse
from sparse import COO
from sparse.utils import assert_eq, random_value_array


x_d = np.array([[1,0],[0,-2]])
y_d = np.array([[3,5],[0, 2]])

x = COO.from_numpy(x_d)
y = COO.from_numpy(y_d)

print "\n x"
print x.todense()
print x.data
print x.coords
print "\n y"
print y.todense()
print y.data
print y.coords
z = x+y
print z
print z.todense()
