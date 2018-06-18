import numpy as np
import six

import sparse
from sparse import BDOK
from sparse.utils import assert_eq

def test_random_shape_nnz(shape, block_shape, density):
    s = sparse.brandom(shape, block_shape, density, format='bdok')

    assert isinstance(s, BDOK)

    assert s.shape == shape
    expected_nnz = density * np.prod(shape)
    assert np.floor(expected_nnz) <= s.nnz <= np.ceil(expected_nnz)

def test():
    shape = (2,)
    block_shape = (2,2)
    density = 0.1 
    test_random_shape_nnz(shape, block_shape, density)
