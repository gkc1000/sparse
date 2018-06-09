import numpy as np
import six

import sparse
from sparse import DOK
from sparse.utils import assert_eq

def test_random_shape_nnz(shape, density):
    s = sparse.random(shape, density, format='dok')

    assert isinstance(s, DOK)

    assert s.shape == shape
    expected_nnz = density * np.prod(shape)
    assert np.floor(expected_nnz) <= s.nnz <= np.ceil(expected_nnz)
