from collections import Iterable
from numbers import Integral

import numba
import numpy as np

from ..compatibility import range, zip_longest
from ..slicing import normalize_index
from ..utils import _zero_of_dtype
from ..coo.indexing import _prune_indices, _mask, _compute_mask


def getitem(x, index):
    """
    This function implements the indexing functionality for BCOO.

    The overall algorithm has three steps:

    1. Normalize the index to canonical form. Function: normalize_index
    2. Get the mask, which is a list of integers corresponding to
       the indices in coords/data for the output data. Function: _mask
    3. Transform the coordinates to what they will be in the output.

    Parameters
    ----------
    x : BCOO
        The array to apply the indexing operation on.
    index : {tuple, str}
        The index into the array.
    """
    from .bcore import BCOO

    # If string, this is an index into an np.void Custom dtype.
    if isinstance(index, str):
        return getblock(x, index)

    # Otherwise, convert into a tuple.
    if not isinstance(index, tuple):
        index = (index,)

    # Check if the last index is an ellipsis.
    last_ellipsis = len(index) > 0 and index[-1] is Ellipsis

    # Normalize the index into canonical form.
    index = normalize_index(index, x.shape)

    # zip_longest so things like x[..., None] are picked up.
    if len(index) != 0 and all(ind == slice(0, dim, 1)
                               for ind, dim in zip_longest(index, x.outer_shape)):
        return x

    if x._offset is None:
        x_offset = (0,) * x.ndim
    else:
        x_offset = x._offset
    if x._last_block_shape is None:
        x_last_block_shape = x.block_shape
    else:
        x_last_block_shape = x._last_block_shape

    # Get the mask
    mask = _outer_mask(x.coords, index, x.shape, x.block_shape, x_offset)

    # Get the length of the mask
    if isinstance(mask, slice):
        n = len(range(mask.start, mask.stop, mask.step))
    else:
        n = len(mask)

    coords = []
    shape = []
    block_mask = []
    block_shape = []
    offset = []
    last_block_shape = []
    i = 0
    for ind in index:
        if isinstance(ind, Integral):
            block_mask.append((ind + x_offset[i]) % x.block_shape[i])
            i += 1
        # Add to the shape and transform the coords in the case of a slice.
        elif isinstance(ind, slice):
            if ind.step < 0:
                raise NotImplementedError('Negative step is not supported')
            elif ind.step > 1:
                raise NotImplementedError('Non-continous step is not supported')
            shape.append(len(range(ind.start, ind.stop, ind.step)))
            block_mask.append(slice(None,None))
            block_shape.append(x.block_shape[i])
            offset.append((ind.start + x_offset[i]) % x.block_shape[i])
            last_block_shape.append((ind.stop - 1 + x_offset[i]) % x.block_shape[i] + 1)
            start = (x_offset[i] + ind.start) // x.block_shape[i]
            coords.append((x.coords[i, mask].astype(int) - start))
            i += 1
        elif isinstance(ind, Iterable):
            raise NotImplementedError('Advanced indexing is not yet supported.')
        # Add a dimension for None.
        elif ind is None:
            coords.append(np.zeros(n, dtype=x.coords.dtype))
            shape.append(1)
            block_mask.append(np.newaxis)
            block_shape.append(1)
            offset.append(0)
            last_block_shape.append(1)

    # Join all the transformed coords.
    if coords:
        coords = np.stack(coords, axis=0)
    else:
        # If index result is a scalar, return a 0-d BCOO or
        # a block depending on whether the last index is an ellipsis.
        if last_ellipsis:
            coords = np.empty((0, n), dtype=np.uint8)
        else:  # scalar
            if n != 0:
                return x.data[(mask,)+tuple(block_mask)][0]
            else:
                return _zero_of_dtype(x.dtype)[()]

    block_shape = tuple(block_shape)
    data = x.data[(mask,)+tuple(block_mask)]

    b = BCOO(coords, data, shape=shape, block_shape=block_shape,
             has_duplicates=False, sorted=True)

    if any(i != 0 for i in offset):
        b._offset = tuple(offset)
        raise NotImplementedError("Offset is not supported.")

    if tuple(last_block_shape) != b.block_shape:
        b._last_block_shape = tuple(last_block_shape)
        raise NotImplementedError("Boundary is not supported.")

    return b

def setitem(x, index, value):
    raise NotImplementedError


def getblock(x, index):
    """
    This function implements the functionality to index BCOO blocks.

    Parameters
    ----------
    x : BCOO
        The array to apply the indexing operation on.
    index : {tuple, str}
        The index into the array.
    """
    from .bcore import BCOO

    # If string, this is an index into an np.void Custom dtype.
    if isinstance(index, str):
        data = x.data[index]
        dtype_shape = x.data.dtype[index].shape
        coords = x.coords
        shape = x.shape
        block_shape = x.block_shape

        if dtype_shape:
            loc0 = np.zeros(coords.shape[1], dtype=coords.dtype)
            coords = np.vstack([coords, (loc0,)*len(dtype_shape)])
            shape = shape + dtype_shape
            block_shape = block_shape + dtype_shape

        b = BCOO(coords, data,
                 shape=shape, block_shape=block_shape,
                 has_duplicates=False,
                 sorted=True)

        if dtype_shape:
            if x._offset is not None:
                b._offset = x._offset + (0,)*len(dtype_shape)
            if x._last_block_shape is not None:
                b._last_block_shape = x._last_block_shape + dtype_shape

        return b

    # Otherwise, convert into a tuple.
    if not isinstance(index, tuple):
        index = (index,)

    # Check if the last index is an ellipsis.
    last_ellipsis = len(index) > 0 and index[-1] is Ellipsis

    # Normalize the index into canonical form.
    index = normalize_index(index, x.outer_shape)

    # zip_longest so things like x[..., None] are picked up.
    if len(index) != 0 and all(ind == slice(0, dim, 1)
                               for ind, dim in zip_longest(index, x.outer_shape)):
        return x

    # Get the mask
    mask = _mask(x.coords, index, x.outer_shape)

    # Get the length of the mask
    if isinstance(mask, slice):
        n = len(range(mask.start, mask.stop, mask.step))
    else:
        n = len(mask)

    coords = []
    outer_shape = []
    block_shape = []
    i = 0
    for ind in index:
        if isinstance(ind, Integral):
            outer_shape.append(1)
            block_shape.append(x.block_shape[i])
            coords.append(np.zeros(n, dtype=x.coords.dtype))
            i += 1
        # Add to the shape and transform the coords in the case of a slice.
        elif isinstance(ind, slice):
            outer_shape.append(len(range(ind.start, ind.stop, ind.step)))
            block_shape.append(x.block_shape[i])
            coords.append((x.coords[i, mask].astype(int) - ind.start) // ind.step)
            i += 1
        elif isinstance(ind, Iterable):
            raise NotImplementedError('Advanced indexing is not yet supported.')
        # Add a dimension for None.
        elif ind is None:
            coords.append(np.zeros(n, dtype=x.coords.dtype))
            outer_shape.append(1)
            block_shape.append(1)

    # Join all the transformed coords.
    if coords:
        coords = np.stack(coords, axis=0)
    else:
        # If index result is a scalar, return a 0-d BCOO or
        # a block depending on whether the last index is an ellipsis.
        if last_ellipsis:
            #raise RuntimeError("bsparse does not support 0-D BCOO")
            coords = np.empty((0, n), dtype=np.uint8)
        else:  # scalar
            if n != 0:
                return x.data[mask][0]
            else:
                return np.zeros((x.block_shape), dtype=x.dtype)

    block_shape = tuple(block_shape)
    shape = tuple([x*y for x, y in zip(outer_shape, block_shape)])
    data = x.data[mask].reshape((-1,) + block_shape)

    return BCOO(coords, data, shape=shape, block_shape=block_shape,
                has_duplicates=False,
                sorted=True)

def setblock(x, index, value):
    raise NotImplementedError


def _outer_mask(coords, indices, shape, block_shape, offset):
    indices = _prune_indices(indices, shape)

    ind_ar = np.empty((len(indices), 3), dtype=np.intp)

    for i, idx in enumerate(indices):
        if isinstance(idx, slice):
            if idx.step != 1:
                raise NotImplementedError('Non-continous step is not supported')
            start = (offset[i] + idx.start) // block_shape[i]
            stop = (offset[i] + idx.stop + block_shape[i]-1) // block_shape[i]
            ind_ar[i] = [start, stop, 1]
        else:  # idx is an integer
            start = (offset[i] + idx) // block_shape[i]
            ind_ar[i] = [start, start + 1, 1]

    mask, is_slice = _compute_mask(coords, ind_ar)

    if is_slice:
        return slice(mask[0], mask[1], 1)
    else:
        return mask

