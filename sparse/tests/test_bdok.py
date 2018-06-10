import numpy as np
import six

import sparse
from sparse import BDOK
from sparse.utils import assert_eq

def test_convert_to_coo():
    s1 = sparse.random((2, 3, 4), 0.5, format='bdok')
    s2 = sparse.BCOO(s1)

    assert_eq(s1, s2)


