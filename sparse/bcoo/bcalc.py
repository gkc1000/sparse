#!/usr/bin/env python
import numpy as np
from sparse import BCOO

# Block sparse version einsum, support contractions between BCOO structures.
# modified from pyscf.lib.einsum

# old version of block einsum
#@profile
def einsum_old(idx_str, *tensors, **kwargs):
    '''Perform a more efficient einsum via reshaping to a matrix multiply.

    Current differences compared to numpy.einsum:
    1. This assumes that each repeated index is actually summed (i.e. no 'i,i->i').
    2. Indices appears only twice (i.e. no 'ij,ik,il->jkl').
    3. The output indices must be explicitly specified (i.e. 'ij,j->i' and not 'ij,j').
    4. No inner contraction (i.e. no 'ijj->i' or 'in,ijj->n').
    5. Indices must overlap (i.e. no 'ij,kl->ijkl').
    6. No ellipsis (i.e. no '...')
    '''

    DEBUG = kwargs.get('DEBUG', False)

    idx_str = idx_str.replace(' ','')
    indices  = idx_str.replace(',','').replace('->','') # avoid use re
    if '->' not in idx_str or any(indices.count(x)>2 for x in set(indices)):
        # TODO 1. No ->, contract over repeated indices 2. more than 2 indices need to contract. 
        raise NotImplementedError

    comma_count = idx_str.count(',')
    if comma_count > 1: # > 2 tensor case
        indices  = idx_str.replace(',',' ').replace('->',' ').split()
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

    elif comma_count == 0: # 1 tensor case
        A = tensors[0]
        idxA, idxC = idx_str.split('->')
        idxA, idxC = [list(x) for x in [idxA, idxC]]
        assert(len(idxA) == A.ndim)
        if isinstance(A, BCOO):
            rangeA = dict(zip(idxA, A.outer_shape))
            block_rangeA = dict(zip(idxA, A.block_shape))
        else:
            return np.einsum(idx_str, A)
        # duplicated indices 'in,ijj->n' # TODO: first index out the repeated indices.
        if len(rangeA) != A.ndim: 
            raise NotImplementedError
        if DEBUG:
            print("*** Einsum for", idx_str)
            print(" idxA =", idxA)
            print(" idxC =", idxC)
        new_orderCt = [idxA.index(idx) for idx in idxC]
        return A.transpose(new_orderCt)

    A, B = tensors
    
    # mix type, transfer to dense case
    if not (isinstance(A, BCOO) and isinstance(B, BCOO)):
        print("Warning: the block einsum takes non-BCOO objects, try to transfer to dense...")
        if hasattr(A, 'todense'):
            A = A.todense()
        if hasattr(B, 'todense'):
            B = B.todense()
        return np.einsum(idx_str, A, B)
            
           
    # ZHC NOTE threshold to determine which lib to use here?        

    # Split the strings into a list of idx char's
    idxA, idxBC = idx_str.split(',')
    idxB, idxC = idxBC.split('->')
    #idxA, idxB, idxC = [list(x) for x in [idxA,idxB,idxC]]
    assert(len(idxA) == A.ndim)
    assert(len(idxB) == B.ndim)

    if DEBUG:
        print("*** Einsum for", idx_str)
        print(" idxA =", idxA)
        print(" idxB =", idxB)
        print(" idxC =", idxC)

    # Get the range for each index and put it in a dictionary
    rangeA = dict(zip(idxA, A.outer_shape))
    rangeB = dict(zip(idxB, B.outer_shape))
    block_rangeA = dict(zip(idxA, A.block_shape))
    block_rangeB = dict(zip(idxB, B.block_shape))
    
    if DEBUG:
        print("rangeA =", rangeA)
        print("rangeB =", rangeB)
        print("block_rangeA =", block_rangeA)
        print("block_rangeB =", block_rangeB)


    # duplicated indices 'in,ijj->n' # TODO: first index out the repeated indices.
    if len(rangeA) != A.ndim or len(rangeB) != B.ndim:
        raise NotImplementedError


    # Find the shared indices being summed over
    shared_idxAB = list(set(idxA).intersection(idxB))
    if len(shared_idxAB) == 0: # TODO Indices must overlap
        raise NotImplementedError

    idxAt = list(idxA)
    idxBt = list(idxB)
    inner_shape = 1
    block_inner_shape = 1
    insert_B_loc = 0
    for n in shared_idxAB:
        if rangeA[n] != rangeB[n]:
            err = ('ERROR: In index string %s, the outer_shape range of index %s is '
                   'different in A (%d) and B (%d)' %
                   (idx_str, n, rangeA[n], rangeB[n]))
            raise ValueError(err)
        if block_rangeA[n] != block_rangeB[n]:
            err = ('ERROR: In index string %s, the block_shape range of index %s is '
                   'different in A (%d) and B (%d)' %
                   (idx_str, n, block_rangeA[n], block_rangeB[n]))
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
    
    if A.nnz == 0 or B.nnz == 0:
        shapeCt = [shapeCt[i] for i in new_orderCt]
        block_shapeCt = [block_shapeCt[i] for i in new_orderCt]
        np_shapeCt = tuple(np.multiply(shapeCt, block_shapeCt))
        
        return BCOO(np.array([],dtype = np.int), data = np.array([], dtype = \
                    np.result_type(A.dtype,B.dtype)), shape=np_shapeCt,\
                    block_shape = block_shapeCt, has_duplicates=False,\
                    sorted=True)

    At = A.transpose(new_orderA)
    Bt = B.transpose(new_orderB)

    # ZHC TODO optimize
    At = At.block_reshape((-1,inner_shape), block_shape = (-1,block_inner_shape))
    Bt = Bt.block_reshape((inner_shape,-1), block_shape = (block_inner_shape,-1))

    #AdotB = At.tobsr().dot(Bt.tobsr())
    At = At.tobsr()
    Bt = Bt.tobsr()
    AdotB = At.dot(Bt)
    AdotB_bcoo = BCOO.from_bsr(AdotB)
    
    if DEBUG:
        print("AdotB bsr format indptr, indices")
        print(AdotB.indptr)
        print(AdotB.indices)
        print("AdotB bcoo format coords")
        print(AdotB_bcoo.coords)
    
    return AdotB_bcoo.block_reshape(shapeCt, block_shape = block_shapeCt).transpose(new_orderCt)



# two tensor contraction
def _contract(subscripts, *tensors, **kwargs):
    DEBUG = kwargs.get('DEBUG', False)

    idx_str = subscripts.replace(' ','')
    indices  = idx_str.replace(',', '').replace('->', '')
    if '->' not in idx_str or any(indices.count(x)>2 for x in set(indices)):
        # TODO 1. No ->, contract over repeated indices 2. more than 2 indices need to contract. 
        raise NotImplementedError

    A, B = tensors
    
    # mix type, transfer to dense case
    if not (isinstance(A, BCOO) and isinstance(B, BCOO)):
        print("Warning: the block einsum takes non-BCOO objects, try to transfer to dense...")
        if hasattr(A, 'todense'):
            A = A.todense()
        if hasattr(B, 'todense'):
            B = B.todense()
        return np.einsum(idx_str, A, B)
            
           
    # ZHC NOTE threshold to determine which lib to use here?        

    # Split the strings into a list of idx char's
    idxA, idxBC = idx_str.split(',')
    idxB, idxC = idxBC.split('->')
    #idxA, idxB, idxC = [list(x) for x in [idxA,idxB,idxC]]
    assert(len(idxA) == A.ndim)
    assert(len(idxB) == B.ndim)

    if DEBUG:
        print("*** Einsum for", idx_str)
        print(" idxA =", idxA)
        print(" idxB =", idxB)
        print(" idxC =", idxC)

    # Get the range for each index and put it in a dictionary
    rangeA = dict(zip(idxA, A.outer_shape))
    rangeB = dict(zip(idxB, B.outer_shape))
    block_rangeA = dict(zip(idxA, A.block_shape))
    block_rangeB = dict(zip(idxB, B.block_shape))
    
    if DEBUG:
        print("rangeA =", rangeA)
        print("rangeB =", rangeB)
        print("block_rangeA =", block_rangeA)
        print("block_rangeB =", block_rangeB)


    # duplicated indices 'in,ijj->n' # TODO: first index out the repeated indices.
    if len(rangeA) != A.ndim or len(rangeB) != B.ndim:
        raise NotImplementedError


    # Find the shared indices being summed over
    shared_idxAB = list(set(idxA).intersection(idxB))
    if len(shared_idxAB) == 0: # TODO Indices must overlap
        raise NotImplementedError

    idxAt = list(idxA)
    idxBt = list(idxB)
    inner_shape = 1
    block_inner_shape = 1
    insert_B_loc = 0
    for n in shared_idxAB:
        if rangeA[n] != rangeB[n]:
            err = ('ERROR: In index string %s, the outer_shape range of index %s is '
                   'different in A (%d) and B (%d)' %
                   (idx_str, n, rangeA[n], rangeB[n]))
            raise ValueError(err)
        if block_rangeA[n] != block_rangeB[n]:
            err = ('ERROR: In index string %s, the block_shape range of index %s is '
                   'different in A (%d) and B (%d)' %
                   (idx_str, n, block_rangeA[n], block_rangeB[n]))
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
    
    if A.nnz == 0 or B.nnz == 0:
        shapeCt = [shapeCt[i] for i in new_orderCt]
        block_shapeCt = [block_shapeCt[i] for i in new_orderCt]
        np_shapeCt = tuple(np.multiply(shapeCt, block_shapeCt))
        
        return BCOO(np.array([],dtype = np.int), data = np.array([], dtype = \
                    np.result_type(A.dtype,B.dtype)), shape=np_shapeCt,\
                    block_shape = block_shapeCt, has_duplicates=False,\
                    sorted=True)

    At = A.transpose(new_orderA)
    Bt = B.transpose(new_orderB)

    At = At.block_reshape((-1,inner_shape), block_shape = (-1,block_inner_shape))
    Bt = Bt.block_reshape((inner_shape,-1), block_shape = (block_inner_shape,-1))

    #AdotB = At.tobsr().dot(Bt.tobsr())
    At = At.tobsr()
    Bt = Bt.tobsr()
    AdotB = At.dot(Bt)
    AdotB_bcoo = BCOO.from_bsr(AdotB)
    
    if DEBUG:
        print("AdotB bsr format indptr, indices")
        print(AdotB.indptr)
        print(AdotB.indices)
        print("AdotB bcoo format coords")
        print(AdotB_bcoo.coords)
    
    return AdotB_bcoo.block_reshape(shapeCt, block_shape = block_shapeCt).transpose(new_orderCt)

# einsum path
if hasattr(np, 'einsum_path'):
    _einsum_path = np.einsum_path
else:
    def _einsum_path(subscripts, *operands, **kwargs):
        #indices  = re.split(',|->', subscripts)
        #indices_in = indices[:-1]
        #idx_final = indices[-1]
        if '->' in subscripts:
            indices_in, idx_final = subscripts.split('->')
            indices_in = indices_in.split(',')
            indices = indices_in + [idx_final]
        else:
            idx_final = ''
            indices_in = subscripts.split('->')[0].split(',')
            indices = indices_in

        if len(indices_in) <= 2:
            idx_removed = set(indices_in[0]).intersection(set(indices_in[1]))
            einsum_str = indices_in[1] + ',' + indices_in[0] + '->' + idx_final
            return operands, [((1,0), idx_removed, einsum_str, idx_final)]

        input_sets = [set(x) for x in indices_in]
        n_shared_max = 0
        for i in range(len(indices_in)):
            for j in range(i):
                tmp = input_sets[i].intersection(input_sets[j])
                n_shared_indices = len(tmp)
                if n_shared_indices > n_shared_max:
                    n_shared_max = n_shared_indices
                    idx_removed = tmp
                    a,b = i,j

        idxA = indices_in.pop(a)
        idxB = indices_in.pop(b)
        rest_idx = ''.join(indices_in) + idx_final
        idx_out = input_sets[a].union(input_sets[b])
        idx_out = ''.join(idx_out.intersection(set(rest_idx)))

        indices_in.append(idx_out)
        einsum_str = idxA + ',' + idxB + '->' + idx_out
        einsum_args = _einsum_path(','.join(indices_in)+'->'+idx_final)[1]
        einsum_args.insert(0, ((a, b), idx_removed, einsum_str, indices_in))
        return operands, einsum_args

#@profile
def einsum(subscripts, *tensors, **kwargs):
    '''Perform a more efficient einsum via reshaping to a matrix multiply.

    Current differences compared to numpy.einsum:
    1. This assumes that each repeated index is actually summed (i.e. no 'i,i->i').
    2. Indices appears only twice (i.e. no 'ij,ik,il->jkl').
    3. The output indices must be explicitly specified (i.e. 'ij,j->i' and not 'ij,j').
    4. No inner contraction (i.e. no 'ijj->i' or 'in,ijj->n').
    5. Indices must overlap (i.e. no 'ij,kl->ijkl').
    6. No ellipsis (i.e. no '...')
    '''
    contract = kwargs.pop('_contract', _contract)

    subscripts = subscripts.replace(' ','')
    if len(tensors) <= 1 or '...' in subscripts:
        #out = np.einsum(subscripts, *tensors, **kwargs)
        raise NotImplementedError
    elif len(tensors) <= 2:
        out = _contract(subscripts, *tensors, **kwargs)
    else:
        if '->' in subscripts:
            indices_in, idx_final = subscripts.split('->')
            indices_in = indices_in.split(',')
        else:
            idx_final = ''
            indices_in = subscripts.split('->')[0].split(',')
        tensors = list(tensors)
        contraction_list = _einsum_path(subscripts, *tensors, optimize=True,
                                        einsum_call=True)[1]
        for contraction in contraction_list:
            inds, idx_rm, einsum_str, remaining = contraction[:4]
            tmp_operands = [tensors.pop(x) for x in inds]
            if len(tmp_operands) > 2:
                #out = np.einsum(einsum_str, *tmp_operands)
                raise NotImplementedError
            else:
                out = contract(einsum_str, *tmp_operands)
            tensors.append(out)
    return out

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
    
    import sparse
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
    


