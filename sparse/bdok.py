from numbers import Integral

import numpy as np

from . import bumpy

from .slicing import normalize_index
from .butils import _zero_of_dtype
from .bsparse_array import BSparseArray
from .compatibility import int, range, zip


class BDOK(BSparseArray):
    """
    A class for building sparse multidimensional arrays.

    Parameters
    ----------
    shape : tuple[int] (BDOK.ndim,)
        The shape of the array.
    data : dict, optional
        The key-value pairs for the data in this array.
    dtype : np.dtype, optional
        The data type of this array. If left empty, it is inferred from
        the first element.

    Attributes
    ----------
    dtype : numpy.dtype
        The datatype of this array. Can be :code:`None` if no elements
        have been set yet.
    shape : tuple[int]
        The shape of this array.
    data : dict
        The keys of this dictionary contain all the indices and the values
        contain the nonzero entries.

    See Also
    --------
    BCOO : A read-only sparse array.

    Examples
    --------
    You can create :obj:`BDOK` objects from Numpy arrays.

    >>> x = np.eye(5, dtype=np.uint8)
    >>> x[2, 3] = 5
    >>> s = BDOK.from_numpy(x)
    >>> s
    <BDOK: shape=(5, 5), dtype=uint8, nnz=6>

    You can also create them from just shapes, and use slicing assignment.

    >>> s2 = BDOK((5, 5), dtype=np.int64)
    >>> s2[1:3, 1:3] = [[4, 5], [6, 7]]
    >>> s2
    <BDOK: shape=(5, 5), dtype=int64, nnz=4>

    You can convert :obj:`BDOK` arrays to :obj:`BCOO` arrays, or :obj:`numpy.ndarray`
    objects.

    >>> from sparse import BCOO
    >>> s3 = BCOO(s2)
    >>> s3
    <BCOO: shape=(5, 5), dtype=int64, nnz=4>
    >>> s2.todense()  # doctest: +NORMALIZE_WHITESPACE
    array([[0, 0, 0, 0, 0],
           [0, 4, 5, 0, 0],
           [0, 6, 7, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])

    >>> s4 = BCOO.from_numpy(np.eye(4, dtype=np.uint8))
    >>> s4
    <BCOO: shape=(4, 4), dtype=uint8, nnz=4>
    >>> s5 = BDOK.from_bcoo(s4)
    >>> s5
    <BDOK: shape=(4, 4), dtype=uint8, nnz=4>

    You can also create :obj:`BDOK` arrays from a shape and a dict of
    values. Zeros are automatically ignored.

    >>> values = {
    ...     (1, 2, 3): 4,
    ...     (3, 2, 1): 0,
    ... }
    >>> s6 = BDOK((5, 5, 5), values)
    >>> s6
    <BDOK: shape=(5, 5, 5), dtype=int64, nnz=1>
    """

    def __init__(self, shape, block_shape=None, data=None, dtype=None):
        from .bcoo import BCOO
        self.data = dict()

        if isinstance(shape, BCOO):
            if block_shape is not None:
                raise RuntimeError('Cannot supply block_shape when converting from BCOO')
            ar = BDOK.from_bcoo(shape)
            self._make_shallow_copy_of(ar)
            return

        if block_shape is None:
            raise RuntimeError('block_shape cannot be None unless initializing from BCOO')
        
        if isinstance(shape, np.ndarray):
            
            ar = BDOK.from_numpy(shape, block_shape)
            self._make_shallow_copy_of(ar)
            return

        self.dtype = np.dtype(dtype)
        super(BDOK, self).__init__(shape, block_shape)

        if not data:
            data = dict()

        if isinstance(data, dict):
            if not dtype:
                if not len(data):
                    self.dtype = np.dtype('float64')
                else:
                    self.dtype = np.result_type(*map(lambda x: np.asarray(x).dtype, data.values()))

            for c, d in data.items():
                self[c] = d
        else:
            raise ValueError('data must be a dict.')

    def _make_shallow_copy_of(self, other):
        super(BDOK, self).__init__(other.shape, other.block_shape)
        self.dtype = other.dtype
        self.data = other.data

    @classmethod
    def from_bcoo(cls, x):
        """
        Get a :obj:`BDOK` array from a :obj:`BCOO` array.

        Parameters
        ----------
        x : BCOO
            The array to convert.

        Returns
        -------
        BDOK
            The equivalent :obj:`BDOK` array.

        Examples
        --------
        >>> from sparse import BCOO
        >>> s = BCOO.from_numpy(np.eye(4))
        >>> s2 = BDOK.from_bcoo(s)
        >>> s2
        <BDOK: shape=(4, 4), dtype=float64, nnz=4>
        """
        ar = cls(x.shape, x.block_shape, dtype=x.dtype)

        for c, d in zip(x.coords.T, x.data):
            ar.data[tuple(c)] = d

        return ar

    def to_bcoo(self):
        """
        Convert this :obj:`BDOK` array to a :obj:`BCOO` array.

        Returns
        -------
        BCOO
            The equivalent :obj:`BCOO` array.

        Examples
        --------
        >>> s = BDOK((5, 5))
        >>> s[1:3, 1:3] = [[4, 5], [6, 7]]
        >>> s
        <BDOK: shape=(5, 5), dtype=float64, nnz=4>
        >>> s2 = s.to_bcoo()
        >>> s2
        <BCOO: shape=(5, 5), dtype=float64, nnz=4>
        """
        from .bcoo import BCOO
        return BCOO(self)

    @classmethod
    def from_numpy(cls, x, block_shape):
        """
        Get a :obj:`BDOK` array from a Numpy array.

        Parameters
        ----------
        x : np.ndarray
            The array to convert.

        Returns
        -------
        BDOK
            The equivalent :obj:`BDOK` array.

        Examples
        --------
        >>> s = BDOK.from_numpy(np.eye(4))
        >>> s
        <BDOK: shape=(4, 4), dtype=float64, nnz=4>
        """
        ar = cls(x.shape, block_shape, dtype=x.dtype)

        # first convert to bndarray
        ba = bumpy.bndarray(ar.outer_shape, block_shape, data = x) 
        sum_x = np.zeros(ar.outer_shape)
        for ix in np.ndindex(sum_x.shape):
            sum_x[ix] = np.sum(np.abs(ba[ix]))
        
        coords = np.nonzero(sum_x)
        data = ba[coords]

        for c in zip(data, *coords):
            d, c = c[0], c[1:]
            ar.data[c] = d

        return ar

    @classmethod
    def from_bumpy(cls, x, block_shape):
        """
        Get a :obj:`BDOK` array from a bumpy array.

        Parameters
        ----------
        x : bumpy.bndarray
            The array to convert.

        Returns
        -------
        BDOK
            The equivalent :obj:`BDOK` array.

        Examples
        --------
        """
        ar = cls(x.shape, block_shape, dtype=x.dtype)

        # first convert to bndarray
        ba = x 
        sum_x = np.zeros(ar.outer_shape)
        for ix in np.ndindex(sum_x.shape):
            sum_x[ix] = np.sum(np.abs(ba[ix]))
        
        coords = np.nonzero(sum_x)
        data = ba[coords]

        for c in zip(data, *coords):
            d, c = c[0], c[1:]
            ar.data[c] = d

        return ar


    @property
    def nnz(self):
        """
        The number of nonzero elements in this array.

        Returns
        -------
        int
            The number of nonzero elements.

        See Also
        --------
        BCOO.nnz : Equivalent :obj:`BCOO` array property.
        numpy.count_nonzero : A similar Numpy function.
        scipy.sparse.bdok_matrix.nnz : The Scipy equivalent property.

        Examples
        --------
        >>> values = {
        ...     (1, 2, 3): 4,
        ...     (3, 2, 1): 0,
        ... }
        >>> s = BDOK((5, 5, 5), values)
        >>> s.nnz
        1
        """
        return len(self.data) * np.product(self.block_shape)

    def __getitem__(self, key):
        key = normalize_index(key, self.outer_shape)

        if not all(isinstance(i, Integral) for i in key):
            raise NotImplementedError('All indices must be integers'
                                      ' when getting an item.')

        if len(key) != self.ndim:
            raise NotImplementedError('Can only get single elements. '
                                      'Expected key of length %d, got %s'
                                      % (self.ndim, str(key)))

        key = tuple(int(k) for k in key)

        if key in self.data:
            return self.data[key]
        else:
            #return _zero_of_dtype(self.dtype)[()]
            return np.zeros(self.block_shape, dtype = self.dtype)

    def __setitem__(self, key, value):
        key = normalize_index(key, self.outer_shape)
        value = np.asanyarray(value)

        value = value.astype(self.dtype)

        key_list = [int(k) if isinstance(k, Integral) else k for k in key]

        self._setitem(key_list, value)

    def _setitem(self, key_list, value):
        #value_missing_dims = len([ind for ind in key_list if isinstance(ind, slice)]) - value.ndim 
        
        # ZHC NOTE: here I think should be some additional treatment of slicing.
        # currently only precise indexing is tested.

        #if value_missing_dims < 0:
        #    raise ValueError('setting an array element with a sequence.')



        for i, ind in enumerate(key_list):
            if isinstance(ind, slice):
                step = ind.step if ind.step is not None else 1
                if step > 0:
                    start = ind.start if ind.start is not None else 0
                    start = max(start, 0)
                    stop = ind.stop if ind.stop is not None else self.outer_shape[i]
                    stop = min(stop, self.outer_shape[i])
                    if start > stop:
                        start = stop
                else:
                    start = ind.start or self.outer_shape[i] - 1
                    stop = ind.stop if ind.stop is not None else -1
                    start = min(start, self.outer_shape[i] - 1)
                    stop = max(stop, -1)
                    if start < stop:
                        start = stop

                key_list_temp = key_list[:]
                for v_idx, ki in enumerate(range(start, stop, step)):
                    key_list_temp[i] = ki
                    vi = value if value_missing_dims > 0 else \
                        (value[0] if value.shape[0] == 1 else value[v_idx])
                    self._setitem(key_list_temp, vi)

                return
            elif not isinstance(ind, Integral):
                raise IndexError('All indices must be slices or integers'
                                 ' when setting an item.')

        key = tuple(key_list)
        #if value != _zero_of_dtype(self.dtype):
        if np.sum(np.abs(value)) != 0.0:
            self.data[key] = value[()]
        elif key in self.data:
            del self.data[key]

    def __str__(self):
        return "<BDOK: shape=%s, outer_shape=%s, block_shape=%s, dtype=%s, nnz=%d>" % (self.shape, self.outer_shape, self.block_shape, self.dtype, self.nnz)

    __repr__ = __str__

    def todense(self):
        """
        Convert this :obj:`BDOK` array into a Numpy array.

        Returns
        -------
        numpy.ndarray
            The equivalent dense array.

        See Also
        --------
        BCOO.todense : Equivalent :obj:`BCOO` array method.
        scipy.sparse.bdok_matrix.todense : Equivalent Scipy method.

        Examples
        --------
        >>> s = BDOK((5, 5))
        >>> s[1:3, 1:3] = [[4, 5], [6, 7]]
        >>> s.todense()  # doctest: +SKIP
        array([[0., 0., 0., 0., 0.],
               [0., 4., 5., 0., 0.],
               [0., 6., 7., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        """
        result = bumpy.zeros(self.outer_shape, self.block_shape, dtype = self.dtype)

        for c, d in self.data.items():
            result[c] = d
        
        return result.todense()

    def to_bumpy(self):
        """
        Convert this :obj:`BDOK` array into a bumpy array.

        Returns
        -------
        bumpy.bndarray
            The equivalent dense array.

        See Also
        --------
        BCOO.todense : Equivalent :obj:`BCOO` array method.
        scipy.sparse.bdok_matrix.todense : Equivalent Scipy method.

        Examples
        --------
        """
        result = bumpy.zeros(self.outer_shape, self.block_shape, dtype = self.dtype)

        for c, d in self.data.items():
            result[c] = d
        
        return result



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
        if format == 'bdok' or format is BDOK:
            return self

        from .bcoo import BCOO
        if format == 'bcoo' or format is BCOO:
            return BCOO.from_iter(self.data, shape=self.shape,
                                  block_shape=self.block_shape)

        raise NotImplementedError('The given format is not supported.')
