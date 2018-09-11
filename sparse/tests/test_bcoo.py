#!/usr/bin/env python
import numpy as np
import six

import sparse
from sparse import BDOK
from sparse import BCOO
from sparse.butils import assert_eq
from sparse.bcoo import bcore

import pytest

def test_brandom():
    x = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5, format='bcoo')
    y = x.todense()
    assert_eq(x, y)

def test_from_numpy():
    #a = np.random.random((6,5,4,1))
    #a = np.zeros((6,5,4,1))
    a = np.array([[1,2,0,0],[0,3,0,0],[4,5,6,0],[8,0,9,0]])
    #x = BCOO.from_numpy(a, block_shape = (2,5,2,1))
    x = BCOO.from_numpy(a, block_shape = (2, 2))
    assert_eq(a, x)

def test_zero_size():
    x = sparse.brandom((0,0,0), (2,2,2))
    assert(x.nnz == 0)
    x = sparse.bcoo.zeros((0,0,0), block_shape=(2,2,2))
    assert(x.nnz == 0)


@pytest.mark.parametrize('shape, dtype, block_shape', [
    [(4,2,4), np.int32, (1,2,2)],
    [(4,4), np.complex128, (1,2)],
    [(4,4), np.float32, (1,2)],
    [(4,4), np.dtype([('a', np.int), ('b', np.float)]), (1,2)],
    [(4, 9, 16), np.dtype('i4,(3,2)f'), (2, 3, 4)],
])
def test_zeros(shape, dtype, block_shape):
    x = sparse.bcoo.zeros(shape, dtype, block_shape)
    assert(x.shape == shape)
    assert(x.block_shape == block_shape)
    assert(x.dtype == dtype)
    assert(x.nnz == 0)
    assert(x.block_nnz == 0)


def test_invalid_shape_error():
    with pytest.raises(RuntimeError):
        sparse.brandom((3, 4), block_shape=(2, 3), format='bcoo')


@pytest.mark.parametrize('axis', [
    None,
    (1, 2, 0),
    (2, 1, 0),
    (0, 1, 2),
    (0, 1, -1),
    (0, -2, -1),
    (-3, -2, -1),
])
def test_transpose(axis):
    x = sparse.brandom((6, 2, 4), (2,2,2), density=0.3)
    y = x.todense()
    xx = x.transpose(axis)
    yy = y.transpose(axis)
    assert_eq(xx, yy)


def test_block_reshape():

    a = np.array([[1, -1, 0, 0], [1 , -1 , 0, 0], [2,3 ,6,7], [4,5,8,9]])
    x = BCOO.from_numpy(a, block_shape = (2,2))
    y = x.todense()

    outer_shape_new = (1,4)
    block_shape_new = (2,2) # unchanged
    z = x.block_reshape(outer_shape_new, block_shape_new)

    print("original matrix (2,2)")
    print(y)
    print("block reshaped matrix (1,4)")
    print(z.todense())


@pytest.mark.parametrize('a, a_bshape, axis, b, b_bshape', [
    #FIXME[(4, 6)      , (2, 3)      , (0, 1)         , (24,)   , (3,)   ],
    [(6, 8)      , (3, 4)      , (0, 1)         , (6, 8)  , (3, 4 )],
    [(6, 8)      , (3, 4)      , (1, 0)         , (8, 6)  , (4, 3 )],
    [(6, 8)      , (3, 4)      , (0, 1)         , (-1, 8) , (3, 4 )],
    [(6, 8)      , (3, 4)      , (1, 0)         , (8, -1) , (4, 3 )],
    #FIXME[(6, 6, 4)   , (2, 3, 4)   , (0, -2, -1)    , (6, 24) , (2, 12)],
    #FIXME[(6, 6, 4)   , (2, 3, 4)   , (0, 2, 1)      , (-1, 6) , (8, 3 )],
    #FIXME[(6, 6, 4)   , (2, 3, 4)   , (2, 1, 0)      , (24, 6) , (12, 2)],
    #FIXME[(6, 6, 4, 5), (2, 3, 4, 5), (0, -2, 3, -3) , (180, 4), (30, 4)],
    #FIXME[(6, 6, 4, 5), (2, 3, 4, 5), (1, 3, 0, 2)   , (30, -1), (15, 8)],
    #FIXME[(6, 6, 4, 5), (2, 3, 4, 5), (2, 1,-4, 3)   , (-1, 5) , (24, 5)],
])
def test_transpose_reshape(a, a_bshape, axis, b, b_bshape):
    x = sparse.brandom(a, a_bshape, density=0.3)
    y = x.todense()
    xx = x.transpose(axis).reshape(b, b_bshape)
    yy = y.transpose(axis).reshape(b)
    assert_eq(xx, yy)


def test_reshape_same():
    s = sparse.bcoo.zeros((4,5,6), block_shape=(2,1,2))

    assert s.reshape(s.shape, s.block_shape) is s


def test_todense():
    s = sparse.bcoo.zeros((4, 9, 16), 'D', block_shape=(2, 3, 4))
    s.todense()

    s = sparse.bcoo.zeros((), block_shape=())
    s.todense()

    s = sparse.bcoo.zeros((4, 9, 16), 'D', block_shape=(2, 3, 4))
    x = s.getblock((1,1,1,Ellipsis))
    x.todense()


#FIXME:@pytest.mark.parametrize('func', [np.expm1, np.log1p, np.sin, np.tan,
#FIXME:                                  np.sinh, np.tanh, np.floor, np.ceil,
#FIXME:                                  np.sqrt, np.conj, np.round, np.rint,
#FIXME:                                  lambda x: x.astype('int32'), np.conjugate,
#FIXME:                                  np.conj, lambda x: x.round(decimals=2), abs])
#FIXME:def test_elemwise(func):
#FIXME:    s = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5)
#FIXME:    x = s.todense()
#FIXME:
#FIXME:    fs = func(s)
#FIXME:    assert isinstance(fs, BCOO)
#FIXME:    assert fs.nnz <= s.nnz
#FIXME:
#FIXME:    assert_eq(func(x), fs)
#FIXME:
#FIXME:
#FIXME:@pytest.mark.parametrize('func', [np.expm1, np.log1p, np.sin, np.tan,
#FIXME:                                  np.sinh, np.tanh, np.floor, np.ceil,
#FIXME:                                  np.sqrt, np.conj,
#FIXME:                                  np.round, np.rint, np.conjugate,
#FIXME:                                  np.conj, lambda x, out: x.round(decimals=2, out=out)])
#FIXME:def test_elemwise_inplace(func):
#FIXME:    s = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5)
#FIXME:    x = s.todense()
#FIXME:
#FIXME:    func(s, out=s)
#FIXME:    func(x, out=x)
#FIXME:    assert isinstance(s, BCOO)
#FIXME:
#FIXME:    assert_eq(x, s)
#FIXME:
#FIXME:
#FIXME:def test_concatenate():
#FIXME:    xx = sparse.random((2, 3, 4), density=0.5)
#FIXME:    x = xx.todense()
#FIXME:    yy = sparse.random((5, 3, 4), density=0.5)
#FIXME:    y = yy.todense()
#FIXME:    zz = sparse.random((4, 3, 4), density=0.5)
#FIXME:    z = zz.todense()
#FIXME:
#FIXME:    assert_eq(np.concatenate([x, y, z], axis=0),
#FIXME:              sparse.concatenate([xx, yy, zz], axis=0))
#FIXME:
#FIXME:    xx = sparse.random((5, 3, 1), density=0.5)
#FIXME:    x = xx.todense()
#FIXME:    yy = sparse.random((5, 3, 3), density=0.5)
#FIXME:    y = yy.todense()
#FIXME:    zz = sparse.random((5, 3, 2), density=0.5)
#FIXME:    z = zz.todense()
#FIXME:
#FIXME:    assert_eq(np.concatenate([x, y, z], axis=2),
#FIXME:              sparse.concatenate([xx, yy, zz], axis=2))
#FIXME:
#FIXME:    assert_eq(np.concatenate([x, y, z], axis=-1),
#FIXME:              sparse.concatenate([xx, yy, zz], axis=-1))
#FIXME:
#FIXME:
#FIXME:@pytest.mark.parametrize('shape', [(5,), (2, 3, 4), (5, 2)])
#FIXME:@pytest.mark.parametrize('axis', [0, 1, -1])
#FIXME:def test_stack(shape, axis):
#FIXME:    xx = sparse.random(shape, density=0.5)
#FIXME:    x = xx.todense()
#FIXME:    yy = sparse.random(shape, density=0.5)
#FIXME:    y = yy.todense()
#FIXME:    zz = sparse.random(shape, density=0.5)
#FIXME:    z = zz.todense()
#FIXME:
#FIXME:    assert_eq(np.stack([x, y, z], axis=axis),
#FIXME:              sparse.stack([xx, yy, zz], axis=axis))
#FIXME:
#FIXME:
#FIXME:def test_addition():
#FIXME:    a = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5)
#FIXME:    x = a.todense()
#FIXME:
#FIXME:    b = sparse.brandom((4, 2, 6), (2, 1, 2), 0.5)
#FIXME:    y = b.todense()
#FIXME:
#FIXME:    assert_eq(x + y, a + b)
#FIXME:    assert_eq(x - y, a - b)
#FIXME:
#FIXME:
#FIXME:def test_scipy_sparse_interface():
#FIXME:    n = 100
#FIXME:    m = 10
#FIXME:    row = np.random.randint(0, n, size=n, dtype=np.uint16)
#FIXME:    col = np.random.randint(0, m, size=n, dtype=np.uint16)
#FIXME:    data = np.ones(n, dtype=np.uint8)
#FIXME:
#FIXME:    inp = (data, (row, col))
#FIXME:
#FIXME:    x = scipy.sparse.coo_matrix(inp)
#FIXME:    xx = BCOO(inp)
#FIXME:
#FIXME:    assert_eq(x, xx, check_nnz=False)
#FIXME:    assert_eq(x.T, xx.T, check_nnz=False)
#FIXME:    assert_eq(xx.to_scipy_sparse(), x, check_nnz=False)
#FIXME:    assert_eq(BCOO.from_scipy_sparse(xx.to_scipy_sparse()), xx, check_nnz=False)
#FIXME:
#FIXME:    assert_eq(x, xx, check_nnz=False)
#FIXME:    assert_eq(x.T.dot(x), xx.T.dot(xx), check_nnz=False)
#FIXME:    assert isinstance(x + xx, BCOO)
#FIXME:    assert isinstance(xx + x, BCOO)


def test_create_with_lists_of_tuples():
    L = [((0, 0, 0), np.random.random((2,4,3))),
         ((1, 2, 1), np.random.random((2,4,3))),
         ((1, 1, 1), np.random.random((2,4,3))),
         ((1, 3, 2), np.random.random((2,4,3)))]

    s = BCOO(L, block_shape=(2,4,3))

    x = np.zeros((2, 4, 3, 2, 4, 3))
    for ind, value in L:
        x[ind] = value
    x = x.transpose(0,3,1,4,2,5).reshape(2*2, 4*4, 3*3)

    assert_eq(s, x)


def test_len():
    s = sparse.brandom((20, 30, 40), block_shape=(2, 3, 4))
    assert len(s) == 20


def test_density():
    s = sparse.brandom((20, 30, 40), block_shape=(2, 3, 4), density=0.1)
    assert np.isclose(s.density, 0.1)


def test_size():
    s = sparse.brandom((20, 30, 40), block_shape=(2, 3, 4))
    assert s.size == 20 * 30 * 40


def test_np_array():
    s = sparse.random((20, 30, 40))
    x = np.array(s)
    assert isinstance(x, np.ndarray)
    assert_eq(x, s)


def test_sizeof():
    import sys
    x = np.eye(100)
    y = BCOO.from_numpy(x, block_shape=(5,5))
    nb = sys.getsizeof(y)
    assert 400 < nb < x.nbytes / 10


def test_tobsr():
    data = np.arange(1,7).repeat(4).reshape((-1,2,2))
    coords = np.array([[0,0,0,2,1,2],[0,1,1,0,2,2]])
    block_shape = (2,2)
    shape = (8,6)
    x = BCOO(coords, data=data, shape=shape, block_shape=block_shape) 
    y = x.todense()
    z = x.tobsr()
    assert_eq(z, y)

def test_invalid_data_input():
    data = np.array([[-1,-2.5],[-3,-4]])
    coords = np.array([[0,1],[0,1]])
    block_shape = (2,2)
    shape = (4,4)
    with pytest.raises(AssertionError):
        x = BCOO(coords, data=data, shape=shape, block_shape=block_shape) 

def test_to_coo():
    a = np.random.random((6,5,4,1))
    a[a > 0.3] = 0.0
    #a = np.zeros((6,5,4,1))
    x = BCOO.from_numpy(a, block_shape = (2,5,2,1))
    from sparse import COO
    y = COO.from_numpy(a)
    z = x.to_coo()
    assert_eq(y,z)

def test_from_coo():
    a = np.random.random((6,5,4,1))
    a[a > 0.3] = 0.0
    #a = np.zeros((6,5,4,1))
    #a = np.array([[1,2,0,0],[0,3,0,0],[4,5,6,0],[8,0,9,0]])
    from sparse import COO
    x = COO.from_numpy(a)
    y = BCOO.from_coo(x, block_shape = (2,5,2,1))
    #y = BCOO.from_coo(x, block_shape = (2,2))
    assert_eq(x,y)

def test_broadcast():
    a = np.array([[1,2,0,0],[0,3,0,0],[4,5,6,0],[8,0,9,0]])
    b = np.broadcast_to(a, (3,4,4))
    x = BCOO.from_numpy(a, block_shape = (2, 2))
    y = x.broadcast_to((3,4,4),(3,2,2))
    assert_eq(b,y)
    
    a = np.random.random((6,5,1,4,1))
    a[a > 0.3] = 0.0
    b = np.broadcast_to(a, (4,6,5,3,4,4))
    x = BCOO.from_numpy(a, block_shape = (3, 5, 1, 2, 1))
    y = x.broadcast_to((4,6,5,3,4,4),(2,3,5,3,2,1))
    assert_eq(b,y)

def test_get_connected_component():
    x = sparse.brandom((15, 8), (3, 2), 0.2, format='bcoo')
    #x += x.T
    y = x.todense()
    
    group_collect =  bcore.get_connected_component(x, sym = False)
    
    sub_coords, sub_offsets, sub_shapes = bcore.index_full2sub(group_collect)
    group_collect_convert_back = bcore.index_sub2full(sub_coords, sub_offsets) 
    assert(group_collect == group_collect_convert_back) 

def test_get_connected_component2():
    x = sparse.brandom((15, 8), (3, 2), 0.2, format='bcoo')
    
    group_collect =  bcore.get_connected_component(x, sym = False)
    clusters = bcore.get_clusters_nosym(x.coords, x.outer_shape)
    print group_collect
    print clusters
    
    group_collect =  bcore.get_connected_component(x, sym = True)
    clusters = bcore.get_clusters(x.coords, x.outer_shape)
    print group_collect
    print clusters

def test_block_eigh2():
    def is_diagonal(A):
        return np.allclose(A - np.diag(np.diagonal(A)), 0.0)
    #np.set_printoptions(3, linewidth = 1000, suppress = True)
    #np.random.seed(0)
    x = sparse.brandom((16, 16), (4, 4), 0.2, format='bcoo')
    x += x.T

    # ZHC TODO
    # support symmetric case with non-square block_shape
    #x = x.to_coo()
    #x = BCOO.from_coo(x, block_shape = (4,2)) 
    
    y = x.todense()
    #eigval_sp, eigvec_sp = bcore.block_eigh(x)[1].todense()
    eigval_sp, eigvec_sp = bcore.block_eigh_(x)

    eigval_sp = eigval_sp.todense()
    eigvec_sp = eigvec_sp.todense()
    
    diagonalized_mat_sp = eigvec_sp.T.dot(x.todense().dot(eigvec_sp))
    #print bcore.block_eigh(x)[1]

    assert(is_diagonal(diagonalized_mat_sp))
    eigval_np = np.linalg.eigh(x.todense())[0]
    assert(np.allclose(np.sort(np.diagonal(diagonalized_mat_sp)), eigval_np))

    print "FINISHED"
    
def test_block_eigh():
    
    def is_diagonal(A):
        return np.allclose(A - np.diag(np.diagonal(A)), 0.0)
    #np.set_printoptions(3, linewidth = 1000, suppress = True)
    #np.random.seed(0)
    x = sparse.brandom((16, 16), (4, 4), 0.2, format='bcoo')
    x += x.T

    # ZHC TODO
    # support symmetric case with non-square block_shape
    #x = x.to_coo()
    #x = BCOO.from_coo(x, block_shape = (4,2)) 
    
    y = x.todense()
    #eigval_sp, eigvec_sp = bcore.block_eigh(x)[1].todense()
    eigval_sp, eigvec_sp = bcore.block_eigh(x, block_sort = True)
    #print eigval_sp
    #print eigvec_sp
    eigval_sp = eigval_sp.todense()
    eigvec_sp = eigvec_sp.todense()
    
    diagonalized_mat_sp = eigvec_sp.T.dot(x.todense().dot(eigvec_sp))
    #print bcore.block_eigh(x)[1]

    assert(is_diagonal(diagonalized_mat_sp))
    eigval_np = np.linalg.eigh(x.todense())[0]
    assert(np.allclose(np.sort(np.diagonal(diagonalized_mat_sp)), eigval_np))

def test_block_eigh_():
    
    def is_diagonal(A):
        return np.allclose(A - np.diag(np.diagonal(A)), 0.0)
    #np.set_printoptions(3, linewidth = 1000, suppress = True)
    #np.random.seed(0)
    x = sparse.brandom((16, 16), (4, 4), 0.2, format='bcoo')
    x += x.T

    # ZHC TODO
    # support symmetric case with non-square block_shape
    #x = x.to_coo()
    #x = BCOO.from_coo(x, block_shape = (4,2)) 
    
    y = x.todense()
    #eigval_sp, eigvec_sp = bcore.block_eigh(x)[1].todense()
    eigval_sp, eigvec_sp = bcore.block_eigh_(x, block_sort = True)
    #print eigval_sp
    #print eigvec_sp
    eigval_sp = eigval_sp.todense()
    eigvec_sp = eigvec_sp.todense()
    
    diagonalized_mat_sp = eigvec_sp.T.dot(x.todense().dot(eigvec_sp))
    #print bcore.block_eigh(x)[1]

    assert(is_diagonal(diagonalized_mat_sp))
    eigval_np = np.linalg.eigh(x.todense())[0]
    assert(np.allclose(np.sort(np.diagonal(diagonalized_mat_sp)), eigval_np))

def test_block_svd():
    
    np.set_printoptions(3, linewidth = 1000, suppress = True)
    x = sparse.brandom((16, 8), (2, 2), 0.3, format='bcoo')
    '''
    a = np.zeros((8, 4))
    a[0:2, 2:4] = np.arange(1, 5).reshape((2, 2))
    a[4:8, 0:2] = np.arange(1, 9).reshape((4, 2))
    x = BCOO.from_numpy(a, block_shape = (2, 2)) 
    '''
    y = x.todense()
    print "original mat"
    print y
    u, sigma, vt = bcore.block_svd(x)
    print sigma
    u_d = u.todense()
    vt_d = vt.todense()
    
    u_np, sigma_np, vt_np = np.linalg.svd(y, full_matrices = True)

    print "u sp"
    print u_d
    print "vt sp"
    print vt_d
    print sigma_np
    print np.sqrt(y.T.dot(y).dot(vt_np.T)/ (vt_np.T))
    print "u np"
    print u_np
    print "vt np"
    print vt_np
    print "uT A v"
    print u_d.T.dot(y.dot(vt_d.T)) 


def test_eigh2():
    x = sparse.brandom((6, 6), (2, 2), 0.2, format='bcoo')
    y = x + x.T
    print y.todense()
    

if __name__ == '__main__':
    print("\n main test \n")
    test_block_eigh2()
    #test_block_svd()
    exit()
    test_brandom()
    test_from_numpy()
    test_transpose(None)
    test_block_reshape()
    test_tobsr()
    test_invalid_data_input()
    test_to_coo()
    test_from_coo()
    test_broadcast()
    test_get_connected_component()
    test_block_eigh()
