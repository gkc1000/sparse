from __future__ import absolute_import, division, print_function
from abc import ABCMeta, abstractmethod
from collections import Iterable
from numbers import Integral
import numpy as np
from .sparse_array import SparseArray

class BSparseArray(SparseArray):
    """
    An abstract base class for all the sparse array classes.

    Attributes
    ----------
    dtype : numpy.dtype
        The data type of this array.
    """

    __metaclass__ = ABCMeta

    def __init__(self, shape, block_shape):
        SparseArray.__init__(self, shape)
        
        if not isinstance(block_shape, Iterable):
            block_shape = (block_shape,)

        if not all(isinstance(l, Integral) and int(l) >= 0 for l in block_shape):
            raise ValueError('block_shape must be an non-negative integer or a tuple '
                             'of non-negative integers.')

        self.block_shape = tuple(int(l) for l in block_shape)
        

        outer_shape, mod_shape = np.divmod(shape, block_shape)
        outer_shape = tuple(outer_shape)

        if np.any(mod_shape) != 0:
            raise RuntimeError('Shape and block_shape are not multiples')
        self.outer_shape=outer_shape
