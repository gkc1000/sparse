#!/usr/bin/env python
import numpy as np
import re
import sparse
from sparse import BCOO
from sparse.utils import assert_eq

# Copied from pyscf.lib.einsum,
# to avoid importing tblis_einsum
# in pyscf.lib.numpy_helper.py
def einsum(idx_str, *tensors, **kwargs):
    '''Perform a more efficient einsum via reshaping to a matrix multiply.

    Current differences compared to numpy.einsum:
    This assumes that each repeated index is actually summed (i.e. no 'i,i->i')
    and appears only twice (i.e. no 'ij,ik,il->jkl'). The output indices must
    be explicitly specified (i.e. 'ij,j->i' and not 'ij,j').
    '''

    DEBUG = kwargs.get('DEBUG', False)

    idx_str = idx_str.replace(' ','')
    indices  = "".join(re.split(',|->',idx_str))
    if '->' not in idx_str or any(indices.count(x)>2 for x in set(indices)):
        #return np.einsum(idx_str,*tensors)
        raise NotImplementedError

    if idx_str.count(',') > 1:
        indices  = re.split(',|->',idx_str)
        indices_in = indices[:-1]
        idx_final = indices[-1]
        n_shared_max = 0
        for i in range(len(indices_in)):
            for j in range(i):
                tmp = list(set(indices_in[i]).intersection(indices_in[j]))
                n_shared_indices = len(tmp)
                if n_shared_indices > n_shared_max:
                    n_shared_max = n_shared_indices
                    shared_indices = tmp
                    [a,b] = [i,j]
        tensors = list(tensors)
        A, B = tensors[a], tensors[b]
        idxA, idxB = indices[a], indices[b]
        idx_out = list(idxA+idxB)
        idx_out = "".join([x for x in idx_out if x not in shared_indices])
        C = einsum(idxA+","+idxB+"->"+idx_out, A, B)
        indices_in.pop(a)
        indices_in.pop(b)
        indices_in.append(idx_out)
        tensors.pop(a)
        tensors.pop(b)
        tensors.append(C)
        return einsum(",".join(indices_in)+"->"+idx_final,*tensors)

    A, B = tensors
    
    # Call numpy.asarray because A or B may be HDF5 Datasets 
    # A = numpy.asarray(A, order='A')
    # B = numpy.asarray(B, order='A')
    # if A.size < 2000 or B.size < 2000:
    #     return numpy.einsum(idx_str, *tensors)

    # Split the strings into a list of idx char's
    idxA, idxBC = idx_str.split(',')
    idxB, idxC = idxBC.split('->')
    idxA, idxB, idxC = [list(x) for x in [idxA,idxB,idxC]]
    assert(len(idxA) == A.ndim)
    assert(len(idxB) == B.ndim)

    if DEBUG:
        print("*** Einsum for", idx_str)
        print(" idxA =", idxA)
        print(" idxB =", idxB)
        print(" idxC =", idxC)

    # Get the range for each index and put it in a dictionary
    rangeA = dict()
    rangeB = dict()
    block_rangeA = dict()
    block_rangeB = dict()
    
    for idx,rnge in zip(idxA,A.outer_shape): # ZHC NOTE
        rangeA[idx] = rnge
    for idx,rnge in zip(idxB,B.outer_shape):
        rangeB[idx] = rnge
    for idx,rnge in zip(idxA,A.block_shape):
        block_rangeA[idx] = rnge
    for idx,rnge in zip(idxB,B.block_shape):
        block_rangeB[idx] = rnge

        
    if DEBUG:
        print("rangeA =", rangeA)
        print("rangeB =", rangeB)
        print("block_rangeA =", block_rangeA)
        print("block_rangeB =", block_rangeB)

    # Find the shared indices being summed over
    shared_idxAB = list(set(idxA).intersection(idxB))
    #if len(shared_idxAB) == 0:
    #    return np.einsum(idx_str,A,B)
    idxAt = list(idxA)
    idxBt = list(idxB)
    inner_shape = 1
    block_inner_shape = 1
    insert_B_loc = 0
    for n in shared_idxAB:
        if rangeA[n] != rangeB[n]:
            err = ('ERROR: In index string %s, the range of index %s is '
                   'different in A (%d) and B (%d)' %
                   (idx_str, n, rangeA[n], rangeB[n]))
            raise RuntimeError(err)

        # Bring idx all the way to the right for A
        # and to the left (but preserve order) for B
        idxA_n = idxAt.index(n)
        idxAt.insert(len(idxAt)-1, idxAt.pop(idxA_n))

        idxB_n = idxBt.index(n)
        idxBt.insert(insert_B_loc, idxBt.pop(idxB_n))
        insert_B_loc += 1

        inner_shape *= rangeA[n]
        block_inner_shape *= block_rangeA[n]

    if DEBUG:
        print("shared_idxAB =", shared_idxAB)
        print("inner_shape =", inner_shape)
        print("block_inner_shape =", block_inner_shape)

    # Transpose the tensors into the proper order and reshape into matrices
    new_orderA = [idxA.index(idx) for idx in idxAt]
    new_orderB = [idxB.index(idx) for idx in idxBt]


    if DEBUG:
        print("Transposing A as", new_orderA)
        print("Transposing B as", new_orderB)
        print("Reshaping A as (-1,", inner_shape, ")")
        print("Reshaping B as (", inner_shape, ",-1)")
        print("Reshaping block A as (-1,", block_inner_shape, ")")
        print("Reshaping block B as (", block_inner_shape, ",-1)")

    shapeCt = list()
    block_shapeCt = list()
    idxCt = list()
    for idx in idxAt:
        if idx in shared_idxAB:
            break
        shapeCt.append(rangeA[idx])
        block_shapeCt.append(block_rangeA[idx])
        idxCt.append(idx)
    for idx in idxBt:
        if idx in shared_idxAB:
            continue
        shapeCt.append(rangeB[idx])
        block_shapeCt.append(block_rangeB[idx])
        idxCt.append(idx)
    new_orderCt = [idxCt.index(idx) for idx in idxC]

    np_shapeCt = tuple(np.multiply(shapeCt, block_shapeCt))
    if A.nnz == 0 or B.nnz == 0:
        shapeCt = [shapeCt[i] for i in new_orderCt]
        block_shapeCt = [block_shapeCt[i] for i in new_orderCt]
        return BCOO(np.array([],dtype = np.int), data = np.array([], dtype = \
                    np.result_type(A.dtype,B.dtype)), shape=np_shapeCt,\
                    block_shape = block_shapeCt, has_duplicates=False,\
                    sorted=True).transpose(new_orderCt)

    At = A.transpose(new_orderA)
    Bt = B.transpose(new_orderB)

    # ZHC TODO optimize
    # if At.flags.f_contiguous:
    #     At = numpy.asarray(At.reshape((-1,inner_shape), (-1,block_inner_shape)), order='F')
    # else:
    At = At.block_reshape((-1,inner_shape), block_shape = (-1,block_inner_shape))
    # if Bt.flags.f_contiguous:
    #     Bt = numpy.asarray(Bt.reshape((inner_shape,-1), (block_inner_shape,-1)), order='F')
    # else:
    Bt = Bt.block_reshape((inner_shape,-1), block_shape = (block_inner_shape,-1))

    AdotB = At.tobsr().dot(Bt.tobsr())
    
    AdotB_bcoo = BCOO.from_bsr(AdotB)
    
    if DEBUG:
        print("AdotB bsr format indptr, indices")
        print(AdotB.indptr)
        print(AdotB.indices)
        print("AdotB bcoo format coords")
        print(AdotB_bcoo.coords)
    
    return AdotB_bcoo.block_reshape(shapeCt, block_shape = block_shapeCt).transpose(new_orderCt)

#def dot(a, b, alpha=1, c=None, beta=0):
#    ab_shape = (a.shape[0], b.shape[1])
#    ab_block_shape = (a.block_shape[0], b.block_shape[1])
#                           
#    ab = bumpy.zeros(ab_shape, ab_block_shape)
#    
#    for i in range(a.shape[0]):
#        for j in range(a.shape[1]):
#            for k in range(b.shape[1]):
#                ab[i,k] += np.dot(a[i,j],b[j,k]) * alpha
#
#    if c is None:
#        c = ab
#    else:
#        if beta == 0:
#            c[:] = 0
#        else:
#            c *= beta
#        c += ab
#    return c


if __name__ == '__main__':

    data_x = np.arange(1,7).repeat(4).reshape((-1,2,2))
    coords_x = np.array([[0,0,0,2,1,2],[0,1,1,0,2,2]])
    shape_x = (8,6)
    block_shape_x = (2,2)
    x = BCOO(coords_x, data=data_x, shape=shape_x, block_shape=block_shape_x)
    x_d = x.todense()
    
    shape_y = (8,4,6)
    block_shape_y = (2,2,2)
    y = sparse.brandom(shape_y, block_shape_y, 0.5, format='bcoo')
    y_d = y.todense()

    c= einsum("ij,ikj->k", x, y, DEBUG = True)
    elemC = np.einsum("ij,ikj->k", x_d, y_d)

    # another test
    '''
    shape_x = (8,4,9)
    block_shape_x = (1,2,3)
    x = sparse.brandom(shape_x, block_shape_x, 0.2, format='bcoo')
    x_d = x.todense()
    
    shape_y = (9,4,6)
    block_shape_y = (3,2,2)
    y = sparse.brandom(shape_y, block_shape_y, 0.5, format='bcoo')
    y_d = y.todense()
    
    
    c= einsum("ijk,kmn->ijmn", x, y, DEBUG = True)
    elemC = np.einsum("ijk,kmn->ijmn", x_d, y_d)
    '''

    print("results comparision:")
    #print(c)
    #print(c.todense())
    print(np.allclose(elemC, c.todense()))
    


