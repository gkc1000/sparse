import numpy as np
from numbers import Integral
from collections import Iterable

def brandom(
        shape,
        block_shape,
        density=0.01,
        random_state=None,
        data_rvs=None,
        format='bcoo'
):
    """ Generate a random sparse multidimensional array

    Parameters
    ----------
    shape: Tuple[int]
        Shape of the array
    density: float, optional
        Density of the generated array.
    random_state : Union[numpy.random.RandomState, int], optional
        Random number generator or random seed. If not given, the
        singleton numpy.random will be used. This random state will be used
        for sampling the sparsity structure, but not necessarily for sampling
        the values of the structurally nonzero entries of the matrix.
    data_rvs : Callable
        Data generation callback. Must accept one single parameter: number of
        :code:`nnz` elements, and return one single NumPy array of exactly
        that length.
    format: str
        The format to return the output array in.

    Returns
    -------
    SparseArray
        The generated random matrix.

    See Also
    --------
    :obj:`scipy.sparse.rand`
        Equivalent Scipy function.
    :obj:`numpy.random.rand`
        Similar Numpy function.

    Examples
    --------

    >>> from sparse import random
    >>> from scipy import stats
    >>> rvs = lambda x: stats.poisson(25, loc=10).rvs(x, random_state=np.random.RandomState(1))
    >>> s = random((2, 3, 4), density=0.25, random_state=np.random.RandomState(1), data_rvs=rvs)
    >>> s.todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  0,  0,  0],
            [ 0, 34,  0,  0],
            [33, 34,  0, 29]],
    <BLANKLINE>
           [[30,  0,  0, 34],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]]])

    """
    # Copied, in large part, from scipy.sparse.random
    # See https://github.com/scipy/scipy/blob/master/LICENSE.txt
    from .bcoo import BCOO

    outer_shape, mod_shape = np.divmod(shape, block_shape)

    elements = np.prod(outer_shape)

    outer_shape = tuple(outer_shape)

    nnz = int(elements * density)

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, Integral):
        random_state = np.random.RandomState(random_state)
    if data_rvs is None:
        data_rvs = random_state.rand

    # Use the algorithm from python's random.sample for k < mn/3.
    if elements < 3 * nnz:
        ind = random_state.choice(elements, size=nnz, replace=False)
    else:
        ind = np.empty(nnz, dtype=np.min_scalar_type(elements - 1))
        selected = set()
        for i in range(nnz):
            j = random_state.randint(elements)
            while j in selected:
                j = random_state.randint(elements)
            selected.add(j)
            ind[i] = j

    data = data_rvs(nnz,*block_shape)

    blk_ind = np.asarray(np.unravel_index(ind, outer_shape))

    ar = BCOO(blk_ind, data, shape, block_shape=block_shape).block_reshape(outer_shape, block_shape)

    return ar.asformat(format)

