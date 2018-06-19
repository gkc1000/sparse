from collections import Iterable, Iterator, Sized, defaultdict, deque
from numbers import Integral

import numpy as np
import scipy.sparse
from numpy.lib.mixins import NDArrayOperatorsMixin

from ..bumpy import bumpy

from .bcommon import dot
from .bindexing import getitem, setitem, getblock, setblock
from .bumath import elemwise, broadcast_to
from ..compatibility import int, range
#from ..sparse_array import SparseArray
from ..bsparse_array import BSparseArray
from ..utils import _zero_of_dtype, normalize_axis


class BCOO(BSparseArray, NDArrayOperatorsMixin):
    """
    A sparse multidimensional array.

    This is stored in COO format.  It depends on NumPy and Scipy.sparse for
    computation, but supports arrays of arbitrary dimension.

    Parameters
    ----------
    coords : numpy.ndarray (COO.ndim, COO.nnz)
        An array holding the index locations of every value
        Should have shape (number of dimensions, number of non-zeros).
    data : numpy.ndarray (COO.nnz,)
        An array of Values. A scalar can also be supplied if the data is the same across
        all coordinates. If not given, defers to :obj:`as_coo`.
    shape : tuple[int] (COO.ndim,)
        The shape of the array.
    has_duplicates : bool, optional
        A value indicating whether the supplied value for :code:`coords` has
        duplicates. Note that setting this to `False` when :code:`coords` does have
        duplicates may result in undefined behaviour. See :obj:`COO.sum_duplicates`
    sorted : bool, optional
        A value indicating whether the values in `coords` are sorted. Note
        that setting this to `False` when :code:`coords` isn't sorted may
        result in undefined behaviour. See :obj:`COO.sort_indices`.
    cache : bool, optional
        Whether to enable cacheing for various operations. See
        :obj:`COO.enable_caching`

    Attributes
    ----------
    coords : numpy.ndarray (ndim, nnz)
        An array holding the coordinates of every nonzero element.
    data : numpy.ndarray (nnz,)
        An array holding the values corresponding to :obj:`COO.coords`.
    shape : tuple[int] (ndim,)
        The dimensions of this array.

    See Also
    --------
    DOK : A mostly write-only sparse array.
    as_coo : Convert any given format to :obj:`COO`.

    Examples
    --------
    You can create :obj:`COO` objects from Numpy arrays.

    >>> x = np.eye(4, dtype=np.uint8)
    >>> x[2, 3] = 5
    >>> s = COO.from_numpy(x)
    >>> s
    <COO: shape=(4, 4), dtype=uint8, nnz=5>
    >>> s.data  # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 1, 5, 1], dtype=uint8)
    >>> s.coords  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 1, 2, 2, 3],
           [0, 1, 2, 3, 3]], dtype=uint8)

    :obj:`COO` objects support basic arithmetic and binary operations.

    >>> x2 = np.eye(4, dtype=np.uint8)
    >>> x2[3, 2] = 5
    >>> s2 = COO.from_numpy(x2)
    >>> (s + s2).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[2, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 2, 5],
           [0, 0, 5, 2]], dtype=uint8)
    >>> (s * s2).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]], dtype=uint8)

    Binary operations support broadcasting.

    >>> x3 = np.zeros((4, 1), dtype=np.uint8)
    >>> x3[2, 0] = 1
    >>> s3 = COO.from_numpy(x3)
    >>> (s * s3).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 1, 5],
           [0, 0, 0, 0]], dtype=uint8)

    :obj:`COO` objects also support dot products and reductions.

    >>> s.dot(s.T).sum(axis=0).todense()   # doctest: +NORMALIZE_WHITESPACE
    array([ 1,  1, 31,  6], dtype=uint64)

    You can use Numpy :code:`ufunc` operations on :obj:`COO` arrays as well.

    >>> np.sum(s, axis=1).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([1, 1, 6, 1], dtype=uint64)
    >>> np.round(np.sqrt(s, dtype=np.float64), decimals=1).todense()   # doctest: +SKIP
    array([[ 1. ,  0. ,  0. ,  0. ],
           [ 0. ,  1. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  2.2],
           [ 0. ,  0. ,  0. ,  1. ]])

    Operations that will result in a dense array will raise a :obj:`ValueError`,
    such as the following.

    >>> np.exp(s)
    Traceback (most recent call last):
        ...
    ValueError: Performing this operation would produce a dense result: <ufunc 'exp'>

    You can also create :obj:`COO` arrays from coordinates and data.

    >>> coords = [[0, 0, 0, 1, 1],
    ...           [0, 1, 2, 0, 3],
    ...           [0, 3, 2, 0, 1]]
    >>> data = [1, 2, 3, 4, 5]
    >>> s4 = COO(coords, data, shape=(3, 4, 5))
    >>> s4
    <COO: shape=(3, 4, 5), dtype=int64, nnz=5>

    If the data is same across all coordinates, you can also specify a scalar.

    >>> coords = [[0, 0, 0, 1, 1],
    ...           [0, 1, 2, 0, 3],
    ...           [0, 3, 2, 0, 1]]
    >>> data = 1
    >>> s5 = COO(coords, data, shape=(3, 4, 5))
    >>> s5
    <COO: shape=(3, 4, 5), dtype=int64, nnz=5>

    Following scipy.sparse conventions you can also pass these as a tuple with
    rows and columns

    >>> rows = [0, 1, 2, 3, 4]
    >>> cols = [0, 0, 0, 1, 1]
    >>> data = [10, 20, 30, 40, 50]
    >>> z = COO((data, (rows, cols)))
    >>> z.todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[10,  0],
           [20,  0],
           [30,  0],
           [ 0, 40],
           [ 0, 50]])

    You can also pass a dictionary or iterable of index/value pairs. Repeated
    indices imply summation:

    >>> d = {(0, 0, 0): 1, (1, 2, 3): 2, (1, 1, 0): 3}
    >>> COO(d)
    <COO: shape=(2, 3, 4), dtype=int64, nnz=3>
    >>> L = [((0, 0), 1),
    ...      ((1, 1), 2),
    ...      ((0, 0), 3)]
    >>> COO(L).todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[4, 0],
           [0, 2]])

    You can convert :obj:`DOK` arrays to :obj:`COO` arrays.

    >>> from sparse import DOK
    >>> s6 = DOK((5, 5), dtype=np.int64)
    >>> s6[1:3, 1:3] = [[4, 5], [6, 7]]
    >>> s6
    <DOK: shape=(5, 5), dtype=int64, nnz=4>
    >>> s7 = s6.asformat('coo')
    >>> s7
    <COO: shape=(5, 5), dtype=int64, nnz=4>
    >>> s7.todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0, 0, 0],
           [0, 4, 5, 0, 0],
           [0, 6, 7, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    """
    __array_priority__ = 12

    def __init__(self, coords, data=None, shape=None, block_shape=None,
                 has_duplicates=True,
                 sorted=False, cache=False):
        self._cache = None
        if cache:
            self.enable_caching()

        if data is None:
            arr = as_bcoo(coords, shape=shape, block_shape=block_shape)
            self._make_shallow_copy_of(arr)
            return

        self.data = np.asarray(data)
        self.coords = np.asarray(coords)

        if self.coords.ndim == 1:
            self.coords = self.coords[None, :]

        if self.data.ndim == 0:
            self.data = np.broadcast_to(self.data, [self.coords.shape[1]] + list(block_shape)) # ZHC NOTE block_shape should be list?


        if shape and not self.coords.size:
            self.coords = np.zeros((len(shape), 0), dtype=np.uint64)

        # default block_shape, a tuple with all 1.
        if block_shape is None:
            if self.coords.nbytes:
                _outer_shape = tuple((self.coords.max(axis=1) + 1)) # is actually outer_shape here
                block_shape = tuple([1] * len(_outer_shape))
            else:
                block_shape = ()

        if shape is None:
            if self.coords.nbytes:
                _outer_shape = tuple((self.coords.max(axis=1) + 1)) # is actually outer_shape here
                shape = tuple(np.asarray(_outer_shape) * np.asarray(block_shape)) # real shape
            else:
                shape = ()

        self._offset = None

        self._last_block_shape = None


        super(BCOO, self).__init__(shape, block_shape)
        #BSparseArray.__init__(self, shape, block_shape)
        if self.shape:
            dtype = np.min_scalar_type(max(max(self.shape) - 1, 0))
        else:
            dtype = np.uint8
        self.coords = self.coords.astype(dtype)
        assert not self.shape or (len(self.data) == self.coords.shape[1] and
                                  len(self.shape) == self.coords.shape[0])

        if not sorted:
            self._sort_indices()

        if has_duplicates:
            self._sum_duplicates()
        
        self.has_canonical_format = True # ZHC NOTE currently we always assume the format is canonical

    def _make_shallow_copy_of(self, other):
        self.coords = other.coords
        self.data = other.data
        super(BCOO, self).__init__(other.shape, other.block_shape)

    def enable_caching(self):
        """ Enable caching of reshape, transpose, and tocsr/csc operations

        This enables efficient iterative workflows that make heavy use of
        csr/csc operations, such as tensordot.  This maintains a cache of
        recent results of reshape and transpose so that operations like
        tensordot (which uses both internally) store efficiently stored
        representations for repeated use.  This can significantly cut down on
        computational costs in common numeric algorithms.

        However, this also assumes that neither this object, nor the downstream
        objects will have their data mutated.

        Examples
        --------
        >>> s.enable_caching()  # doctest: +SKIP
        >>> csr1 = s.transpose((2, 0, 1)).reshape((100, 120)).tocsr()  # doctest: +SKIP
        >>> csr2 = s.transpose((2, 0, 1)).reshape((100, 120)).tocsr()  # doctest: +SKIP
        >>> csr1 is csr2  # doctest: +SKIP
        True
        """
        self._cache = defaultdict(lambda: deque(maxlen=3))

    @classmethod
    def from_numpy(cls, x, block_shape):
        """
        Convert the given :obj:`numpy.ndarray` to a :obj:`COO` object.

        Parameters
        ----------
        x : np.ndarray
            The dense array to convert.

        Returns
        -------
        COO
            The converted COO array.

        Examples
        --------
        >>> x = np.eye(5)
        >>> s = BCOO.from_numpy(x)
        >>> s
        <COO: shape=(5, 5), dtype=float64, nnz=5>
        """
        x = np.asanyarray(x)

        if x.shape:
            if not isinstance(block_shape, Iterable):
                block_shape = (block_shape,)

            if not all(isinstance(l, Integral) and int(l) >= 0 for l in block_shape):
                raise ValueError('block_shape must be an non-negative integer or a tuple '
                                 'of non-negative integers.')

            block_shape = tuple(int(l) for l in block_shape)

            outer_shape, mod_shape = np.divmod(x.shape, block_shape)
            outer_shape = tuple(outer_shape)

            if np.any(mod_shape) != 0:
                raise RuntimeError('Shape and block_shape are not multiples')

            # first convert to bndarray
            ba = bumpy.bndarray(shape = outer_shape, block_shape = block_shape, data = x)
            sum_x = np.zeros(outer_shape)
            for ix in np.ndindex(sum_x.shape):
                sum_x[ix] = np.sum(np.abs(ba[ix]))

            coords = np.nonzero(sum_x)
            data = np.asarray(list(ba[coords]))
            coords = np.vstack(coords)
        else:
            coords = np.empty((0, 1), dtype=np.uint8)
            data = np.array(x, ndmin=1)

        return cls(coords, data, shape=x.shape, block_shape = block_shape, has_duplicates=False,
                   sorted=True)

    def todense(self):
        """
        Convert this :obj:`COO` array to a dense :obj:`numpy.ndarray`. Note that
        this may take a large amount of memory if the :obj:`COO` object's :code:`shape`
        is large.

        Returns
        -------
        numpy.ndarray
            The converted dense array.

        See Also
        --------
        DOK.todense : Equivalent :obj:`DOK` array method.
        scipy.sparse.coo_matrix.todense : Equivalent Scipy method.

        Examples
        --------
        >>> x = np.random.randint(100, size=(7, 3))
        >>> s = BCOO.from_numpy(x)
        >>> x2 = s.todense()
        >>> np.array_equal(x, x2)
        True
        """

        if not (self._offset is None and self._last_block_shape is None):
            raise NotImplementedError("Offset and boundary are not supported.")

        result = bumpy.zeros(self.outer_shape, self.block_shape, dtype = self.dtype)

        data = self.data
        

        if self.coords is not None:
            for i in range(self.coords.shape[1]):
                result[tuple(self.coords[:, i])] = data[i]
        else:
            if len(data) != 0:
                result[coords] = data

        return result.todense()

    @classmethod
    def from_scipy_sparse(cls, x): # ZHC TODO change to BCOO
        """
        Construct a :obj:`COO` array from a :obj:`scipy.sparse.spmatrix`

        Parameters
        ----------
        x : scipy.sparse.spmatrix
            The sparse matrix to construct the array from.

        Returns
        -------
        COO
            The converted :obj:`COO` object.

        Examples
        --------
        >>> x = scipy.sparse.rand(6, 3, density=0.2)
        >>> s = BCOO.from_scipy_sparse(x)
        >>> np.array_equal(x.todense(), s.todense())
        True
        """
        x = x.asformat('coo')
        coords = np.empty((2, x.nnz), dtype=x.row.dtype)
        coords[0, :] = x.row
        coords[1, :] = x.col
        return COO(coords, x.data, shape=x.shape,
                   has_duplicates=not x.has_canonical_format,
                   sorted=x.has_canonical_format)

    @classmethod
    def from_iter(cls, x, shape=None, block_shape=None):
        """
        Converts an iterable in certain formats to a :obj:`COO` array. See examples
        for details.

        Parameters
        ----------
        x : Iterable or Iterator
            The iterable to convert to :obj:`COO`.
        shape : tuple[int], optional
            The shape of the array.

        Returns
        -------
        out : COO
            The output :obj:`COO` array.

        Examples
        --------
        You can convert items of the format ``[((i, j, k), value), ((i, j, k), value)]`` to :obj:`COO`.
        Here, the first part represents the coordinate and the second part represents the value.

        >>> x = [((0, 0), 1), ((1, 1), 1)]
        >>> s = BCOO.from_iter(x)
        >>> s.todense()
        array([[1, 0],
               [0, 1]])

        You can also have a similar format with a dictionary.

        >>> x = {(0, 0): 1, (1, 1): 1}
        >>> s = BCOO.from_iter(x)
        >>> s.todense()
        array([[1, 0],
               [0, 1]])

        The third supported format is ``(data, (..., row, col))``.

        >>> x = ([1, 1], ([0, 1], [0, 1]))
        >>> s = BCOO.from_iter(x)
        >>> s.todense()
        array([[1, 0],
               [0, 1]])

        You can also pass in a :obj:`collections.Iterator` object.

        >>> x = [((0, 0), 1), ((1, 1), 1)].__iter__()
        >>> s = BCOO.from_iter(x)
        >>> s.todense()
        array([[1, 0],
               [0, 1]])
        """
        if isinstance(x, dict):
            x = list(x.items())

        if not isinstance(x, Sized):
            x = list(x)

        if len(x) != 2 and not all(len(item) == 2 for item in x):
            raise ValueError('Invalid iterable to convert to BCOO.')

        if not x:
            ndim = 0 if shape is None else len(shape)
            coords = np.empty((ndim, 0), dtype=np.uint8)
            data = np.empty((0,))

            return BCOO(coords, data, shape=() if shape is None else shape, block_shape=() if block_shape is None else block_shape,
                        sorted=True, has_duplicates=False)

        if not isinstance(x[0][0], Iterable):
            coords = np.stack(x[1], axis=0)
            data = np.asarray(x[0])
        else:
            coords = np.array([item[0] for item in x]).T
            data = np.array([item[1] for item in x])

        if not (coords.ndim == 2 and np.issubdtype(coords.dtype, np.integer) and np.all(coords >= 0)):
            raise ValueError('Invalid iterable to convert to BCOO.')
        if block_shape is not None and data.shape[1:] != block_shape:
            raise ValueError('Invalid iterable to convert to BCOO.')

        return BCOO(coords, data, shape=shape, block_shape=block_shape)

    @property
    def dtype(self):
        """
        The datatype of this array.

        Returns
        -------
        numpy.dtype
            The datatype of this array.

        See Also
        --------
        numpy.ndarray.dtype : Numpy equivalent property.
        scipy.sparse.coo_matrix.dtype : Scipy equivalent property.

        Examples
        --------
        >>> x = (200 * np.random.rand(5, 4)).astype(np.int32)
        >>> s = BCOO.from_numpy(x)
        >>> s.dtype
        dtype('int32')
        >>> x.dtype == s.dtype
        True
        """
        return self.data.dtype

    @property
    def nnz(self):
        """
        The number of nonzero elements in this array.

        Returns
        -------
        int
            The number of nonzero elements in this array.

        See Also
        --------
        DOK.nnz : Equivalent :obj:`DOK` array property.
        numpy.count_nonzero : A similar Numpy function.
        scipy.sparse.coo_matrix.nnz : The Scipy equivalent property.

        Examples
        --------
        >>> x = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 0])
        >>> np.count_nonzero(x)
        6
        >>> s = BCOO.from_numpy(x)
        >>> s.nnz
        6
        >>> np.count_nonzero(x) == s.nnz
        True
        """
        return self.coords.shape[1] * np.product(self.block_shape)

    @property
    def block_nnz(self):
        """
        The number of nonzero blocks. Note that any duplicates in
        :code:`coords` are counted multiple times. To avoid this, call :obj:`BCOO.sum_duplicates`.

        Returns
        -------
        int
            The number of non-zero blocks.
        """
        return self.coords.shape[1]

    @property
    def nbytes(self):
        """
        The number of bytes taken up by this object. Note that for small arrays,
        this may undercount the number of bytes due to the large constant overhead.

        Returns
        -------
        int
            The approximate bytes of memory taken by this object.

        See Also
        --------
        numpy.ndarray.nbytes : The equivalent Numpy property.

        Examples
        --------
        >>> data = np.arange(6, dtype=np.uint8)
        >>> coords = np.random.randint(1000, size=(3, 6), dtype=np.uint16)
        >>> s = COO(coords, data, shape=(1000, 1000, 1000))
        >>> s.nbytes
        42
        """
        return self.data.nbytes + self.coords.nbytes

    def __len__(self):
        """
        Get "length" of array, which is by definition the size of the first
        dimension.

        Returns
        -------
        int
            The size of the first dimension.

        See Also
        --------
        numpy.ndarray.__len__ : Numpy equivalent property.

        Examples
        --------
        >>> x = np.zeros((10, 10))
        >>> s = BCOO.from_numpy(x)
        >>> len(s)
        10
        """
        return self.shape[0]

    def __sizeof__(self):
        return self.nbytes

    __getitem__ = getitem
    __setitem__ = setitem

    getblock = getblock
    setblock = setblock

    def __str__(self):
        return "<BCOO: shape=%s, block_shape=%s, dtype=%s, nnz=%d, block_nnz=%d>" % (self.shape, self.block_shape, self.dtype, self.nnz, self.block_nnz)

    __repr__ = __str__

    @staticmethod
    def _reduce(method, *args, **kwargs):
        assert len(args) == 1

        self = args[0]
        if isinstance(self, scipy.sparse.spmatrix):
            self = BCOO.from_scipy_sparse(self)

        return self.reduce(method, **kwargs)

    def reduce(self, method, axis=(0,), keepdims=False, **kwargs):
        """
        Performs a reduction operation on this array.

        Parameters
        ----------
        method : numpy.ufunc
            The method to use for performing the reduction.
        axis : Union[int, Iterable[int]], optional
            The axes along which to perform the reduction. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        kwargs : dict
            Any extra arguments to pass to the reduction operation.

        Returns
        -------
        COO
            The result of the reduction operation.

        Raises
        ------
        ValueError
            If reducing an all-zero axis would produce a nonzero result.

        Notes
        -----
        This function internally calls :obj:`BCOO.sum_duplicates` to bring the array into
        canonical form.

        See Also
        --------
        numpy.ufunc.reduce : A similar Numpy method.
        BCOO.nanreduce : Similar method with ``NaN`` skipping functionality.

        Examples
        --------
        You can use the :obj:`BCOO.reduce` method to apply a reduction operation to
        any Numpy :code:`ufunc`.

        >>> x = np.ones((5, 5), dtype=np.int)
        >>> s = BCOO.from_numpy(x)
        >>> s2 = s.reduce(np.add, axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([5, 5, 5, 5, 5])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        reduction.

        >>> s3 = s.reduce(np.add, axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        You can also pass in any keyword argument that :obj:`numpy.ufunc.reduce` supports.
        For example, :code:`dtype`. Note that :code:`out` isn't supported.

        >>> s4 = s.reduce(np.add, axis=1, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array by only the first axis.

        >>> s.reduce(np.add)
        <COO: shape=(5,), dtype=int64, nnz=5>
        """
        axis = normalize_axis(axis, self.ndim)
        zero_reduce_result = method.reduce([_zero_of_dtype(self.dtype)], **kwargs)

        if zero_reduce_result != _zero_of_dtype(np.dtype(zero_reduce_result)):
            raise ValueError("Performing this reduction operation would produce "
                             "a dense result: %s" % str(method))

        if axis is None:
            axis = tuple(range(self.ndim))

        if not isinstance(axis, tuple):
            axis = (axis,)

        axis = tuple(a if a >= 0 else a + self.ndim for a in axis)

        if set(axis) == set(range(self.ndim)):
            result = method.reduce(self.data, **kwargs)
            if self.nnz != self.size:
                result = method(result, _zero_of_dtype(self.dtype)[()], **kwargs)
        else:
            axis = tuple(axis)
            neg_axis = tuple(ax for ax in range(self.ndim) if ax not in set(axis))

            a = self.transpose(neg_axis + axis)
            a = a.reshape((np.prod([self.shape[d] for d in neg_axis]),
                           np.prod([self.shape[d] for d in axis])))

            result, inv_idx, counts = _grouped_reduce(a.data, a.coords[0], method, **kwargs)
            missing_counts = counts != a.shape[1]
            result[missing_counts] = method(result[missing_counts],
                                            _zero_of_dtype(self.dtype), **kwargs)
            coords = a.coords[0:1, inv_idx]

            # Filter out zeros
            mask = result != _zero_of_dtype(result.dtype)
            coords = coords[:, mask]
            result = result[mask]

            a = BCOO(coords, result, shape=(a.shape[0],),
                     has_duplicates=False, sorted=True)

            a = a.reshape(tuple(self.shape[d] for d in neg_axis))
            result = a

        if keepdims:
            result = _keepdims(self, result, axis)
        return result

    def sum(self, axis=None, keepdims=False, dtype=None, out=None):
        """
        Performs a sum operation along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to sum. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        COO
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.sum` : Equivalent numpy function.
        scipy.sparse.coo_matrix.sum : Equivalent Scipy function.
        :obj:`nansum` : Function with ``NaN`` skipping.

        Notes
        -----
        * This function internally calls :obj:`BCOO.sum_duplicates` to bring the array into
          canonical form.
        * The :code:`out` parameter is provided just for compatibility with Numpy and
          isn't actually supported.

        Examples
        --------
        You can use :obj:`BCOO.sum` to sum an array across any dimension.

        >>> x = np.ones((5, 5), dtype=np.int)
        >>> s = BCOO.from_numpy(x)
        >>> s2 = s.sum(axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([5, 5, 5, 5, 5])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        sum.

        >>> s3 = s.sum(axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        You can pass in an output datatype, if needed.

        >>> s4 = s.sum(axis=1, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, summing along all axes.

        >>> s.sum()
        25
        """
        return np.add.reduce(self, out=out, axis=axis, keepdims=keepdims, dtype=dtype)

    def max(self, axis=None, keepdims=False, out=None):
        """
        Maximize along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to maximize. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        COO
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.max` : Equivalent numpy function.
        scipy.sparse.coo_matrix.max : Equivalent Scipy function.
        :obj:`nanmax` : Function with ``NaN`` skipping.

        Notes
        -----
        * This function internally calls :obj:`BCOO.sum_duplicates` to bring the array into
          canonical form.
        * The :code:`out` parameter is provided just for compatibility with Numpy and
          isn't actually supported.

        Examples
        --------
        You can use :obj:`BCOO.max` to maximize an array across any dimension.

        >>> x = np.add.outer(np.arange(5), np.arange(5))
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6],
               [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8]])
        >>> s = BCOO.from_numpy(x)
        >>> s2 = s.max(axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([4, 5, 6, 7, 8])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        maximization.

        >>> s3 = s.max(axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        By default, this reduces the array down to one number, maximizing along all axes.

        >>> s.max()
        8
        """
        return np.maximum.reduce(self, out=out, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False, out=None):
        """
        Minimize along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to minimize. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        COO
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.min` : Equivalent numpy function.
        scipy.sparse.coo_matrix.min : Equivalent Scipy function.
        :obj:`nanmin` : Function with ``NaN`` skipping.

        Notes
        -----
        * This function internally calls :obj:`BCOO.sum_duplicates` to bring the array into
          canonical form.
        * The :code:`out` parameter is provided just for compatibility with Numpy and
          isn't actually supported.

        Examples
        --------
        You can use :obj:`BCOO.min` to minimize an array across any dimension.

        >>> x = np.add.outer(np.arange(5), np.arange(5))
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6],
               [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8]])
        >>> s = BCOO.from_numpy(x)
        >>> s2 = s.min(axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([0, 1, 2, 3, 4])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        minimization.

        >>> s3 = s.min(axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        By default, this reduces the array down to one number, minimizing along all axes.

        >>> s.min()
        0
        """
        return np.minimum.reduce(self, out=out, axis=axis, keepdims=keepdims)

    def prod(self, axis=None, keepdims=False, dtype=None, out=None):
        """
        Performs a product operation along the given axes. Uses all axes by default.

        Parameters
        ----------
        axis : Union[int, Iterable[int]], optional
            The axes along which to multiply. Uses all axes by default.
        keepdims : bool, optional
            Whether or not to keep the dimensions of the original array.
        dtype: numpy.dtype
            The data type of the output array.

        Returns
        -------
        COO
            The reduced output sparse array.

        See Also
        --------
        :obj:`numpy.prod` : Equivalent numpy function.
        :obj:`nanprod` : Function with ``NaN`` skipping.

        Notes
        -----
        * This function internally calls :obj:`BCOO.sum_duplicates` to bring the array into
          canonical form.
        * The :code:`out` parameter is provided just for compatibility with Numpy and
          isn't actually supported.

        Examples
        --------
        You can use :obj:`BCOO.prod` to multiply an array across any dimension.

        >>> x = np.add.outer(np.arange(5), np.arange(5))
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2, 3, 4],
               [1, 2, 3, 4, 5],
               [2, 3, 4, 5, 6],
               [3, 4, 5, 6, 7],
               [4, 5, 6, 7, 8]])
        >>> s = BCOO.from_numpy(x)
        >>> s2 = s.prod(axis=1)
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([   0,  120,  720, 2520, 6720])

        You can also use the :code:`keepdims` argument to keep the dimensions after the
        reduction.

        >>> s3 = s.prod(axis=1, keepdims=True)
        >>> s3.shape
        (5, 1)

        You can pass in an output datatype, if needed.

        >>> s4 = s.prod(axis=1, dtype=np.float16)
        >>> s4.dtype
        dtype('float16')

        By default, this reduces the array down to one number, multiplying along all axes.

        >>> s.prod()
        0
        """
        return np.multiply.reduce(self, out=out, axis=axis, keepdims=keepdims, dtype=dtype)

    def transpose(self, axes=None):
        """
        Returns a new array which has the order of the axes switched.

        Parameters
        ----------
        axes : Iterable[int], optional
            The new order of the axes compared to the previous one. Reverses the axes
            by default.

        Returns
        -------
        COO
            The new array with the axes in the desired order.

        See Also
        --------
        :obj:`BCOO.T` : A quick property to reverse the order of the axes.
        numpy.ndarray.transpose : Numpy equivalent function.

        Examples
        --------
        We can change the order of the dimensions of any :obj:`COO` array with this
        function.

        >>> x = np.add.outer(np.arange(5), np.arange(5)[::-1])
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[4, 3, 2, 1, 0],
               [5, 4, 3, 2, 1],
               [6, 5, 4, 3, 2],
               [7, 6, 5, 4, 3],
               [8, 7, 6, 5, 4]])
        >>> s = BCOO.from_numpy(x)
        >>> s.transpose((1, 0)).todense()  # doctest: +NORMALIZE_WHITESPACE
        array([[4, 5, 6, 7, 8],
               [3, 4, 5, 6, 7],
               [2, 3, 4, 5, 6],
               [1, 2, 3, 4, 5],
               [0, 1, 2, 3, 4]])

        Note that by default, this reverses the order of the axes rather than switching
        the last and second-to-last axes as required by some linear algebra operations.

        >>> x = np.random.rand(2, 3, 4)
        >>> s = BCOO.from_numpy(x)
        >>> s.transpose().shape
        (4, 3, 2)
        """
        if axes is None:
            axes = list(reversed(range(self.ndim)))

        # Normalize all axes indices to positive values
        axes = normalize_axis(axes, self.ndim)

        if len(np.unique(axes)) < len(axes):
            raise ValueError("repeated axis in transpose")

        if not len(axes) == self.ndim:
            raise ValueError("axes don't match array")

        axes = tuple(axes)

        if axes == tuple(range(self.ndim)):
            return self

        if self._cache is not None: # ZHC NOTE to discuss
            for ax, value in self._cache['transpose']:
                if ax == axes:
                    return value

        shape = tuple(self.shape[ax] for ax in axes)
        block_shape = tuple(self.block_shape[ax] for ax in axes)
        data_trans_axes = np.array([-1] + list(axes)) + 1
        #data_T = np.asarray(list(self.data)).transpose(data_trans_axes) # ZHC NOTE the storage of self.data should be an array of shape (block_nnz, *(block_shape))
        data_T = self.data.transpose(data_trans_axes)

        # there is no duplicates, but is not sorted
        result = BCOO(self.coords[axes, :], data_T, shape, \
                     block_shape = block_shape, has_duplicates=False, \
                     cache=self._cache is not None)

        if self._cache is not None:
            self._cache['transpose'].append((axes, result))
        return result

    @property
    def T(self):
        """
        Returns a new array which has the order of the axes reversed.

        Returns
        -------
        COO
            The new array with the axes in the desired order.

        See Also
        --------
        :obj:`BCOO.transpose` : A method where you can specify the order of the axes.
        numpy.ndarray.T : Numpy equivalent property.

        Examples
        --------
        We can change the order of the dimensions of any :obj:`COO` array with this
        function.

        >>> x = np.add.outer(np.arange(5), np.arange(5)[::-1])
        >>> x  # doctest: +NORMALIZE_WHITESPACE
        array([[4, 3, 2, 1, 0],
               [5, 4, 3, 2, 1],
               [6, 5, 4, 3, 2],
               [7, 6, 5, 4, 3],
               [8, 7, 6, 5, 4]])
        >>> s = BCOO.from_numpy(x)
        >>> s.T.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([[4, 5, 6, 7, 8],
               [3, 4, 5, 6, 7],
               [2, 3, 4, 5, 6],
               [1, 2, 3, 4, 5],
               [0, 1, 2, 3, 4]])

        Note that by default, this reverses the order of the axes rather than switching
        the last and second-to-last axes as required by some linear algebra operations.

        >>> x = np.random.rand(2, 3, 4)
        >>> s = BCOO.from_numpy(x)
        >>> s.T.shape
        (4, 3, 2)
        """
        return self.transpose(tuple(range(self.ndim))[::-1])

    def dot(self, other):
        """
        Performs the equivalent of :code:`x.dot(y)` for :obj:`COO`.

        Parameters
        ----------
        other : Union[COO, numpy.ndarray, scipy.sparse.spmatrix]
            The second operand of the dot product operation.

        Returns
        -------
        {COO, numpy.ndarray}
            The result of the dot product. If the result turns out to be dense,
            then a dense array is returned, otherwise, a sparse array.

        See Also
        --------
        dot : Equivalent function for two arguments.
        :obj:`numpy.dot` : Numpy equivalent function.
        scipy.sparse.coo_matrix.dot : Scipy equivalent function.

        Examples
        --------
        >>> x = np.arange(4).reshape((2, 2))
        >>> s = BCOO.from_numpy(x)
        >>> s.dot(s) # doctest: +SKIP
        array([[ 2,  3],
               [ 6, 11]], dtype=int64)
        """
        return dot(self, other)

    def __matmul__(self, other):
        try:
            return dot(self, other)
        except NotImplementedError:
            return NotImplemented

    def __rmatmul__(self, other):
        try:
            return dot(other, self)
        except NotImplementedError:
            return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop('out', None)
        if out is not None and not all(isinstance(x, BCOO) for x in out):
            return NotImplemented

        if method == '__call__':
            result = elemwise(ufunc, *inputs, **kwargs)
        elif method == 'reduce':
            result = BCOO._reduce(ufunc, *inputs, **kwargs)
        else:
            return NotImplemented

        if out is not None:
            (out,) = out
            if out.shape != result.shape:
                raise ValueError('non-broadcastable output operand with shape %s'
                                 'doesn\'t match the broadcast shape %s' % (out.shape, result.shape))

            out._make_shallow_copy_of(result)
            return out

        return result

    def __array__(self, dtype=None, **kwargs):
        x = self.todense()
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        return x

    def linear_loc(self, signed=False):
        """
        The nonzero coordinates of a flattened version of this array. Note that
        the coordinates may be out of order.

        Parameters
        ----------
        signed : bool, optional
            Whether to use a signed datatype for the output array. :code:`False`
            by default.

        Returns
        -------
        numpy.ndarray
            The flattened coordinates.

        See Also
        --------
        :obj:`numpy.flatnonzero` : Equivalent Numpy function.

        Examples
        --------
        >>> x = np.eye(5)
        >>> s = BCOO.from_numpy(x)
        >>> s.linear_loc()  # doctest: +NORMALIZE_WHITESPACE
        array([ 0,  6, 12, 18, 24], dtype=uint8)
        >>> np.array_equal(np.flatnonzero(x), s.linear_loc())
        True
        """
        from .bcommon import linear_loc

        return linear_loc(self.coords, self.shape, signed)

    def reshape(self, shape, block_shape):
        """
        Returns a new :obj:`COO` array that is a reshaped version of this array.

        Parameters
        ----------
        shape : tuple[int]
            The desired shape of the output array.

        Returns
        -------
        COO
            The reshaped output array.

        See Also
        --------
        numpy.ndarray.reshape : The equivalent Numpy function.

        Examples
        --------
        >>> s = BCOO.from_numpy(np.arange(25))
        >>> s2 = s.reshape((5, 5))
        >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24]])
        """
        if self.shape == shape and self.block_shape == block_shape:
            return self

        if any(d == -1 for d in shape):
            extra = int(self.size /
                        np.prod([d for d in shape if d != -1]))
            shape = tuple([d if d != -1 else extra for d in shape])

        if any(d == -1 for d in block_shape):
            extra = int(self.size /
                        np.prod([d for d in block_shape if d != -1]))
            block_shape = tuple([d if d != -1 else extra for d in block_shape])

        if self.shape == shape and self.block_shape == block_shape:
            return self

        if self._cache is not None:
            for sh, value in self._cache['reshape']:
                if sh == shape:
                    return value

        # TODO: this self.size enforces a 2**64 limit to array size
        linear_loc = self.linear_loc()


        outer_shape, mod_shape = np.divmod(shape, block_shape)
        max_shape = max(outer_shape) if len(outer_shape) != 0 else 1

        coords = np.empty((len(outer_shape), self.block_nnz), dtype=np.min_scalar_type(max_shape - 1))
        strides = 1
        for i, d in enumerate(outer_shape[::-1]):
            coords[-(i + 1), :] = (linear_loc // strides) % d
            strides *= d

        data = np.reshape(self.data, tuple([self.data.shape[0]] + list(block_shape)))

        result = BCOO(coords, data, shape, block_shape,
                     has_duplicates=False,
                     sorted=True, cache=self._cache is not None)

        if self._cache is not None:
            self._cache['reshape'].append((shape, result))
        return result


    def block_reshape(self, outer_shape, block_shape = None):
        """
        Returns a new :obj:`BCOO` array that is a block reshaped version of this array.

        Parameters
        ----------
        outer_shape : tuple[int]
            The desired outer_shape of the output array.
        block_shape : tuple [int]
            The desired block_shape of the output array.
            default is to have same block_shape as the original one.

        Returns
        -------
        BCOO
            The reshaped output array.

        See Also
        --------
        numpy.ndarray.reshape : The equivalent Numpy function.

        Examples
        --------
        """
        if block_shape is None:
            block_shape = self.block_shape


        if any(d == -1 for d in outer_shape):
            extra = int(self.size / int(np.prod(self.block_shape)) /
                        np.prod([d for d in outer_shape if d != -1]))
            outer_shape = tuple([d if d != -1 else extra for d in outer_shape])

        if any(d == -1 for d in block_shape):
            extra = int(self.size / int(np.prod(self.outer_shape)) /
                        np.prod([d for d in block_shape if d != -1]))
            block_shape = tuple([d if d != -1 else extra for d in block_shape])

        if self.outer_shape == outer_shape and self.block_shape == block_shape:
            return self

        if self._cache is not None: # ZHC NOTE to discuss
            for sh, value in self._cache['reshape']:
                if sh == outer_shape:
                    return value

        # TODO: this self.size enforces a 2**64 limit to array size
        linear_loc = self.linear_loc()

        #outer_shape, mod_shape = np.divmod(shape, block_shape)

        # only redetermine the indices, keep the data unchanged. should be efficient.
        idx_1d = np.ravel_multi_index(self.coords, self.outer_shape)
        coords_new = np.asarray(np.unravel_index(idx_1d, outer_shape))

        # reshape the block data only if specifying block_shape
        if block_shape != self.block_shape:
            data = np.asarray([data_i.reshape(block_shape) for i, data_i in enumerate(self.data)]) # TODO: some optimization for memory
        else:
            data = self.data

        shape = tuple(np.multiply(outer_shape, block_shape))

        # ZHC NOTE consider not to copy?
        # already no duplicates and sorted
        result = BCOO(coords_new, data, shape, block_shape,
                     has_duplicates=False,
                     sorted=True, cache=self._cache is not None)

        if self._cache is not None:
            self._cache['reshape'].append((outer_shape, result))
        return result

    def to_scipy_sparse(self):
        """
        Converts this :obj:`COO` object into a :obj:`scipy.sparse.coo_matrix`.

        Returns
        -------
        :obj:`scipy.sparse.coo_matrix`
            The converted Scipy sparse matrix.

        Raises
        ------
        ValueError
            If the array is not two-dimensional.

        See Also
        --------
        BCOO.tocsr : Convert to a :obj:`scipy.sparse.csr_matrix`.
        BCOO.tocsc : Convert to a :obj:`scipy.sparse.csc_matrix`.
        """
        if self.ndim != 2:
            raise ValueError("Can only convert a 2-dimensional array to a Scipy sparse matrix.")

        result = scipy.sparse.coo_matrix((self.data,
                                          (self.coords[0],
                                           self.coords[1])),
                                         shape=self.shape)
        result.has_canonical_format = True
        return result

    def _tocsr(self):
        if self.ndim != 2:
            raise ValueError('This array must be two-dimensional for this conversion '
                             'to work.')
        row, col = self.coords

        # Pass 3: count nonzeros in each row
        indptr = np.zeros(self.shape[0] + 1, dtype=np.int64)
        np.cumsum(np.bincount(row, minlength=self.shape[0]), out=indptr[1:])

        return scipy.sparse.csr_matrix((self.data, col, indptr), shape=self.shape)

    def tocsr(self):
        """
        Converts this array to a :obj:`scipy.sparse.csr_matrix`.

        Returns
        -------
        scipy.sparse.csr_matrix
            The result of the conversion.

        Raises
        ------
        ValueError
            If the array is not two-dimensional.

        See Also
        --------
        BCOO.tocsc : Convert to a :obj:`scipy.sparse.csc_matrix`.
        BCOO.to_scipy_sparse : Convert to a :obj:`scipy.sparse.coo_matrix`.
        scipy.sparse.coo_matrix.tocsr : Equivalent Scipy function.
        """
        if self._cache is not None:
            try:
                return self._csr
            except AttributeError:
                pass
            try:
                self._csr = self._csc.tocsr()
                return self._csr
            except AttributeError:
                pass

            self._csr = csr = self._tocsr()
        else:
            csr = self._tocsr()
        return csr

    def tocsc(self):
        """
        Converts this array to a :obj:`scipy.sparse.csc_matrix`.

        Returns
        -------
        scipy.sparse.csc_matrix
            The result of the conversion.

        Raises
        ------
        ValueError
            If the array is not two-dimensional.

        See Also
        --------
        BCOO.tocsr : Convert to a :obj:`scipy.sparse.csr_matrix`.
        BCOO.to_scipy_sparse : Convert to a :obj:`scipy.sparse.coo_matrix`.
        scipy.sparse.coo_matrix.tocsc : Equivalent Scipy function.
        """
        if self._cache is not None:
            try:
                return self._csc
            except AttributeError:
                pass
            try:
                self._csc = self._csr.tocsc()
                return self._csc
            except AttributeError:
                pass

            self._csc = csc = self.tocsr().tocsc()
        else:
            csc = self.tocsr().tocsc()

        return csc

    @classmethod
    def from_bsr(cls, x):
        """
        Convert the given :obj:`scipy.bsr_matrix` to a :obj:`BCOO` object.

        Parameters
        ----------
        x : scipy.bsr_matrix
            The bsr_matrix to convert.

        Returns
        -------
        BCOO
            The converted BCOO array.

        Examples
        --------
        """
        assert isinstance(x, scipy.sparse.bsr.bsr_matrix)
        
        x.sum_duplicates() # ZHC NOTE necessary?

        shape = x.shape
        block_shape = x.blocksize
        
        indptr_diff = np.diff(x.indptr)
        if indptr_diff.dtype.itemsize > np.dtype(np.intp).itemsize:
            # Check for potential overflow
            indptr_diff_limited = indptr_diff.astype(np.intp)
            if np.any(indptr_diff_limited != indptr_diff):
                raise ValueError("Matrix too big to convert")
            indptr_diff = indptr_diff_limited

        row = (np.arange(shape[0]//block_shape[0])).repeat(indptr_diff)
        col = x.indices
        data = x.data
      
        coords = np.vstack((row, col))

        # already no duplicates and sorted
        return cls(coords, data, shape = shape, block_shape = block_shape, has_duplicates=False,
                   sorted=True)

    def _tobsr(self):
        if self.ndim != 2:
            raise ValueError('This array must be two-dimensional for this conversion '
                             'to work.')
        row, col = self.coords

        # ZHC NOTE : here we assume the coords are sorted and no duplicates. 
        # Pass 3: count nonzeros in each row
        indptr = np.zeros(self.outer_shape[0] + 1, dtype = np.int64)
        np.cumsum(np.bincount(row, minlength = self.outer_shape[0]), out = indptr[1:])

        return scipy.sparse.bsr_matrix((self.data, col, indptr), shape = self.shape, blocksize = self.block_shape)
        

    def tobsr(self):
        """
        Converts this array to a :obj:`scipy.sparse.bsr_matrix`.

        Returns
        -------
        scipy.sparse.bsr_matrix
            The result of the conversion.

        Raises
        ------
        ValueError
            If the array is not two-dimensional.

        See Also
        --------
        BCOO.tocsc : Convert to a :obj:`scipy.sparse.csc_matrix`.
        BCOO.to_scipy_sparse : Convert to a :obj:`scipy.sparse.coo_matrix`.
        """
        # ZHC NOTE currently need the canonical format to convert
        assert(self.has_canonical_format)

        if self._cache is not None:
            try:
                return self._bsr
            except AttributeError:
                pass

            self._bsr = bsr = self._tobsr()
        else:
            bsr = self._tobsr()

        bsr.has_sorted_indices = True
        bsr.has_canonical_format = True

        return bsr


    def _sort_indices(self):
        """
        Sorts the :obj:`BCOO.coords` attribute. Also sorts the data in
        :obj:`BCOO.data` to match.

        Examples
        --------
        >>> coords = np.array([[1, 2, 0]], dtype=np.uint8)
        >>> data = np.array([4, 1, 3], dtype=np.uint8)
        >>> s = COO(coords, data)
        >>> s._sort_indices()
        >>> s.coords  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2]], dtype=uint8)
        >>> s.data  # doctest: +NORMALIZE_WHITESPACE
        array([3, 4, 1], dtype=uint8)
        """
        linear = self.linear_loc(signed=True)

        if (np.diff(linear) > 0).all():  # already sorted
            return

        order = np.argsort(linear)
        self.coords = self.coords[:, order]
        self.data = self.data[order]

    def _sum_duplicates(self):
        """
        Sums data corresponding to duplicates in :obj:`BCOO.coords`.

        See Also
        --------
        scipy.sparse.coo_matrix.sum_duplicates : Equivalent Scipy function.

        Examples
        --------
        >>> coords = np.array([[0, 1, 1, 2]], dtype=np.uint8)
        >>> data = np.array([6, 5, 2, 2], dtype=np.uint8)
        >>> s = COO(coords, data)
        >>> s._sum_duplicates()
        >>> s.coords  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 1, 2]], dtype=uint8)
        >>> s.data  # doctest: +NORMALIZE_WHITESPACE
        array([6, 7, 2], dtype=uint8)
        """
        # Inspired by scipy/sparse/coo.py::sum_duplicates
        # See https://github.com/scipy/scipy/blob/master/LICENSE.txt
        linear = self.linear_loc()
        unique_mask = np.diff(linear) != 0

        if unique_mask.sum() == len(unique_mask):  # already unique
            return

        unique_mask = np.append(True, unique_mask)

        coords = self.coords[:, unique_mask]
        (unique_inds,) = np.nonzero(unique_mask)
        data = np.add.reduceat(self.data, unique_inds, dtype=self.data.dtype)

        self.data = data
        self.coords = coords

    def broadcast_to(self, shape):
        """
        Performs the equivalent of :obj:`numpy.broadcast_to` for :obj:`COO`. Note that
        this function returns a new array instead of a view.

        Parameters
        ----------
        shape : tuple[int]
            The shape to broadcast the data to.

        Returns
        -------
        COO
            The broadcasted sparse array.

        Raises
        ------
        ValueError
            If the operand cannot be broadcast to the given shape.

        See also
        --------
        :obj:`numpy.broadcast_to` : NumPy equivalent function
        """
        return broadcast_to(self, shape)

    def round(self, decimals=0, out=None):
        """
        Evenly round to the given number of decimals.

        See also
        --------
        :obj:`numpy.round` : NumPy equivalent ufunc.
        :obj:`BCOO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        if out is not None and not isinstance(out, tuple):
            out = (out,)
        return self.__array_ufunc__(np.round, '__call__', self, decimals=decimals, out=out)

    def astype(self, dtype):
        """
        Copy of the array, cast to a specified type.

        See also
        --------
        scipy.sparse.coo_matrix.astype : SciPy sparse equivalent function
        numpy.ndarray.astype : NumPy equivalent ufunc.
        :obj:`BCOO.elemwise`: Apply an arbitrary element-wise function to one or two
            arguments.

        Notes
        -----
        The :code:`out` parameter is provided just for compatibility with Numpy and isn't
        actually supported.
        """
        return self.__array_ufunc__(np.ndarray.astype, '__call__', self, dtype=dtype)

    def maybe_densify(self, max_size=1000, min_density=0.25):
        """
        Converts this :obj:`COO` array to a :obj:`numpy.ndarray` if not too
        costly.

        Parameters
        ----------
        max_size : int
            Maximum number of elements in output
        min_density : float
            Minimum density of output

        Returns
        -------
        numpy.ndarray
            The dense array.

        Raises
        -------
        ValueError
            If the returned array would be too large.

        Examples
        --------
        Convert a small sparse array to a dense array.

        >>> s = BCOO.from_numpy(np.random.rand(2, 3, 4))
        >>> x = s.maybe_densify()
        >>> np.allclose(x, s.todense())
        True

        You can also specify the minimum allowed density or the maximum number
        of output elements. If both conditions are unmet, this method will throw
        an error.

        >>> x = np.zeros((5, 5), dtype=np.uint8)
        >>> x[2, 2] = 1
        >>> s = BCOO.from_numpy(x)
        >>> s.maybe_densify(max_size=5, min_density=0.25)
        Traceback (most recent call last):
            ...
        ValueError: Operation would require converting large sparse array to dense
        """
        if self.size <= max_size or self.density >= min_density:
            return self.todense()
        else:
            raise ValueError("Operation would require converting "
                             "large sparse array to dense")

    def nonzero(self):
        """
        Get the indices where this array is nonzero.

        Returns
        -------
        idx : tuple[numpy.ndarray]
            The indices where this array is nonzero.

        See Also
        --------
        :obj:`numpy.ndarray.nonzero` : NumPy equivalent function

        Examples
        --------
        >>> s = BCOO.from_numpy(np.eye(5))
        >>> s.nonzero()
        (array([0, 1, 2, 3, 4], dtype=uint8), array([0, 1, 2, 3, 4], dtype=uint8))
        """
        return tuple(self.coords)

    def asformat(self, format):
        """
        Convert this sparse array to a given format.

        Parameters
        ----------
        format : str
            A format string.

        Returns
        -------
        out : SparseArray
            The converted array.

        Raises
        ------
        NotImplementedError
            If the format isn't supported.
        """
        if format == 'bcoo' or format is BCOO:
            return self

        from ..bdok import BDOK
        if format == 'bdok' or format is BDOK:
            return BDOK.from_bcoo(self)

        raise NotImplementedError('The given format is not supported.')


def as_bcoo(x, shape=None, block_shape=None):
    """
    Converts any given format to :obj:`COO`. See the "See Also" section for details.

    Parameters
    ----------
    x : SparseArray or numpy.ndarray or scipy.sparse.spmatrix or Iterable.
        The item to convert.
    shape : tuple[int], optional
        The shape of the output array. Can only be used in case of Iterable.

    Returns
    -------
    out : COO
        The converted :obj:`COO` array.

    See Also
    --------
    SparseArray.asformat : A utility function to convert between formats in this library.
    BCOO.from_numpy : Convert a Numpy array to :obj:`COO`.
    BCOO.from_scipy_sparse : Convert a SciPy sparse matrix to :obj:`COO`.
    BCOO.from_iter : Convert an iterable to :obj:`COO`.
    """
    if hasattr(x, 'shape') and shape is not None:
        raise ValueError('Cannot provide a shape in combination with something '
                         'that already has a shape.')

    if isinstance(x, BSparseArray):
        return x.asformat('bcoo')

    if isinstance(x, np.ndarray):
        return BCOO.from_numpy(x, block_shape)

    if isinstance(x, scipy.sparse.spmatrix):
        return BCOO.from_scipy_sparse(x)

    if isinstance(x, (Iterable, Iterator)):
        return BCOO.from_iter(x, shape=shape, block_shape=block_shape)

    raise NotImplementedError('Format not supported for conversion. Supplied type is '
                              '%s, see help(sparse.as_coo) for supported formats.' % type(x))


def _keepdims(original, new, axis):
    shape = list(original.shape)
    for ax in axis:
        shape[ax] = 1
    return new.reshape(shape)


def _grouped_reduce(x, groups, method, **kwargs):
    """
    Performs a :code:`ufunc` grouped reduce.

    Parameters
    ----------
    x : np.ndarray
        The data to reduce.
    groups : np.ndarray
        The groups the data belongs to. The groups must be
        contiguous.
    method : np.ufunc
        The :code:`ufunc` to use to perform the reduction.
    kwargs : dict
        The kwargs to pass to the :code:`ufunc`'s :code:`reduceat`
        function.

    Returns
    -------
    result : np.ndarray
        The result of the grouped reduce operation.
    inv_idx : np.ndarray
        The index of the first element where each group is found.
    counts : np.ndarray
        The number of elements in each group.
    """
    # Partial credit to @shoyer
    # Ref: https://gist.github.com/shoyer/f538ac78ae904c936844
    flag = np.concatenate(([True] if len(x) != 0 else [], groups[1:] != groups[:-1]))
    inv_idx = np.flatnonzero(flag)
    result = method.reduceat(x, inv_idx, **kwargs)
    counts = np.diff(np.concatenate((inv_idx, [len(x)])))
    return result, inv_idx, counts
