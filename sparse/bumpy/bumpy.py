#!/usr/bin/env python
import numpy as np
import bumpy_helper

class bndarray(np.ndarray):
    def __new__(subtype, shape, block_shape, data = None, block_dtype = None):
        if len(shape) != len(block_shape):
            raise RuntimeError
        obj = np.ndarray.__new__(subtype, shape, dtype=np.object)
        obj.block_shape = tuple(block_shape)

        if block_dtype is None:
            if data is None:
                block_dtype = float
            else:
                block_dtype = data.dtype
        else:
            if data is not None:
                data = data.astype(block_dtype)

        obj.block_dtype = block_dtype

        if data is None:
            for ix in np.ndindex(obj.shape):
                obj[ix]=np.empty(block_shape, dtype=block_dtype)
        else:
            for ix in np.ndindex(shape):
                start = np.multiply(ix, block_shape)
                end = np.add(start, block_shape)
                slices = [slice(s,e) for (s,e) in zip(start,end)]
                obj[ix] = data[slices]

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        else:
            self.block_shape = obj.block_shape
            self.block_dtype = obj.block_dtype

    def reshape(self, newshape, block_shape=None, order='C'):
        obj=np.ndarray.reshape(self, newshape, order=order)
        if block_shape is not None:
            for ix in np.ndindex(obj.shape):
                obj[ix] = np.reshape(obj[ix], block_shape, order=order)
        return obj
        
    def transpose(self, axes=None):
        obj = self.copy()
        obj = np.ndarray.transpose(obj, axes)
        if axes == None:
            obj.block_shape = self.block_shape[::-1]
        else:
            obj.block_shape = tuple(np.array(self.block_shape)[axes])
        
        for ix in np.ndindex(obj.shape):
            obj[ix] = obj[ix].transpose(axes)
        return obj

    def todense(self):
        return asndarray(self)
        


def zeros(shape, block_shape, dtype=float):
    obj = bndarray(shape, block_shape, dtype)
    for ix in np.ndindex(obj.shape):
        obj[ix] = np.zeros(obj.block_shape, dtype)
    return obj

def eye(N, block_N, dtype=float):
    shape = [N, N]
    block_shape = [block_N, block_N]
    obj = zeros(shape, block_shape, dtype)
    for i in range(obj.shape[0]):
        obj[i,i] = np.eye(block_N, dtype=dtype)
    return obj

# should be in bumpy/random
def random(shape, block_shape, dtype=float):
    obj = bndarray(shape, block_shape, dtype)
    for ix in np.ndindex(obj.shape):
        obj[ix] = np.random.random(obj.block_shape)
    return obj
    
def asndarray(ba):
    shape = np.multiply(ba.shape, ba.block_shape)
    a = np.empty(shape, dtype=ba.block_dtype)
    for ix in np.ndindex(ba.shape):
        start = np.multiply(ix, ba.block_shape)
        end = np.add(start, ba.block_shape)
        slices = [slice(s,e) for (s,e) in zip(start,end)]
        a[slices] = ba[ix]
    return a
    

##################
# test functions #
##################


def test_init_from_numpy():
    #a = np.random.random((6,9,3))
    #shape = (2,3,3)
    #block_shape =(3,3,1)
    a = np.random.random((4,4))
    shape = (2,2)
    block_shape =(2,2)
    ba = bndarray(shape, block_shape, data = a) 
    print "original ndarray"
    print a
    print "bndarray"
    print ba
    print np.allclose(a,asndarray(ba))



def test_einsum():
    print "\nmain program\n"
    #a = bndarray((3,1,2), (2,4,7)) 
    #a = bndarray((1,2), (2,4))
    a = random((3,5), (2,5))
    b = random((5,2), (5,2))

    einsum = bumpy_helper.einsum

    c= einsum("ij,jk->ik", a,b)
    A = asndarray(a)
    B = asndarray(b)
    C = asndarray(c)

    elemC = np.einsum("ij,jk->ik", A, B)

    print np.allclose(elemC, C)
     

if __name__ == '__main__':
    print "\n main test \n"
    #test_einsum()
    test_init_from_numpy()
    
