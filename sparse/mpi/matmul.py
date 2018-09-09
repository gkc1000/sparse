#!/usr/bin/env python
#!/usr/bin/env python

import sys
import time
import threading
import numpy
from mpi4py import MPI
from pyscf import lib


comm = MPI.COMM_WORLD
mpi_size = size = comm.Get_size()
mpi_rank = rank = comm.Get_rank()
INT_MAX = 2147483647
BLKSIZE = INT_MAX // 32 + 1

def _create_dtype(dat):
    mpi_dtype = MPI._typedict[dat.dtype.char]
    # the smallest power of 2 greater than dat.size/INT_MAX
    deriv_dtype_len = 1 << max(0, dat.size.bit_length()-31)
    deriv_dtype = mpi_dtype.Create_contiguous(deriv_dtype_len).Commit()
    count, rest = dat.size.__divmod__(deriv_dtype_len)
    return deriv_dtype, count, rest

def bcast_test(buf, root=0):  # To test, maybe with better performance
    buf = numpy.asarray(buf, order='C')
    shape, dtype = comm.bcast((buf.shape, buf.dtype.char))
    if rank != root:
        buf = numpy.empty(shape, dtype=dtype)

    if buf.size <= BLKSIZE:
        comm.Bcast(buf, root)
    else:
        deriv_dtype, count, rest = _create_dtype(buf)
        comm.Bcast([buf, count, deriv_dtype], root)
        comm.Bcast(buf[-rest*deriv_dtype.size:], root)
    return buf

def bcast(buf, root=0):
    buf = numpy.asarray(buf, order='C')
    shape, dtype = comm.bcast((buf.shape, buf.dtype.char))
    if rank != root:
        buf = numpy.empty(shape, dtype=dtype)

    buf_seg = numpy.ndarray(buf.size, dtype=buf.dtype, buffer=buf)
    for p0, p1 in lib.prange(0, buf.size, BLKSIZE):
        comm.Bcast(buf_seg[p0:p1], root)
    return buf

def reduce(sendbuf, op=MPI.SUM, root=0):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype)

    recvbuf = numpy.zeros_like(sendbuf)
    send_seg = numpy.ndarray(sendbuf.size, dtype=sendbuf.dtype, buffer=sendbuf)
    recv_seg = numpy.ndarray(recvbuf.size, dtype=recvbuf.dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Reduce(send_seg[p0:p1], recv_seg[p0:p1], op, root)

    if rank == root:
        return recvbuf
    else:
        return sendbuf

def allreduce(sendbuf, op=MPI.SUM):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype)

    recvbuf = numpy.zeros_like(sendbuf)
    send_seg = numpy.ndarray(sendbuf.size, dtype=sendbuf.dtype, buffer=sendbuf)
    recv_seg = numpy.ndarray(recvbuf.size, dtype=recvbuf.dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Allreduce(send_seg[p0:p1], recv_seg[p0:p1], op)
    return recvbuf

def scatter(sendbuf, root=0):
    if rank == root:
        mpi_dtype = numpy.result_type(*sendbuf).char
        shape = comm.scatter([x.shape for x in sendbuf])
        counts = numpy.asarray([x.size for x in sendbuf])
        comm.bcast((mpi_dtype, counts))
        sendbuf = [numpy.asarray(x, mpi_dtype).ravel() for x in sendbuf]
        sendbuf = numpy.hstack(sendbuf)
    else:
        shape = comm.scatter(None)
        mpi_dtype, counts = comm.bcast(None)

    displs = numpy.append(0, numpy.cumsum(counts[:-1]))
    recvbuf = numpy.empty(numpy.prod(shape), dtype=mpi_dtype)

    #DONOT use lib.prange. lib.prange may terminate early in some processes
    for p0, p1 in prange(0, numpy.max(counts), BLKSIZE):
        counts_seg = _segment_counts(counts, p0, p1)
        comm.Scatterv([sendbuf, counts_seg, displs+p0, mpi_dtype],
                      [recvbuf[p0:p1], mpi_dtype], root)
    return recvbuf.reshape(shape)

def gather(sendbuf, root=0, split_recvbuf=False):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape = sendbuf.shape
    size_dtype = comm.allgather((shape, sendbuf.dtype.char))
    rshape = [x[0] for x in size_dtype]
    counts = numpy.array([numpy.prod(x) for x in rshape])

    mpi_dtype = numpy.result_type(*[x[1] for x in size_dtype]).char
    _assert(sendbuf.dtype == mpi_dtype or sendbuf.size == 0)

    if rank == root:
        displs = numpy.append(0, numpy.cumsum(counts[:-1]))
        recvbuf = numpy.empty(sum(counts), dtype=mpi_dtype)

        sendbuf = sendbuf.ravel()
        for p0, p1 in lib.prange(0, numpy.max(counts), BLKSIZE):
            counts_seg = _segment_counts(counts, p0, p1)
            comm.Gatherv([sendbuf[p0:p1], mpi_dtype],
                         [recvbuf, counts_seg, displs+p0, mpi_dtype], root)
        if split_recvbuf:
            return [recvbuf[p0:p0+c].reshape(shape)
                    for p0,c,shape in zip(displs,counts,rshape)]
        else:
            try:
                return recvbuf.reshape((-1,) + shape[1:])
            except ValueError:
                return recvbuf
    else:
        send_seg = sendbuf.ravel()
        for p0, p1 in lib.prange(0, numpy.max(counts), BLKSIZE):
            comm.Gatherv([send_seg[p0:p1], mpi_dtype], None, root)
        return sendbuf


def allgather(sendbuf, split_recvbuf=False):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape = sendbuf.shape
    attr = comm.allgather((shape, sendbuf.dtype.char))
    rshape = [x[0] for x in attr]
    counts = numpy.array([numpy.prod(x) for x in rshape])
    mpi_dtype = numpy.result_type(*[x[1] for x in attr]).char
    _assert(sendbuf.dtype.char == mpi_dtype or sendbuf.size == 0)

    displs = numpy.append(0, numpy.cumsum(counts[:-1]))
    recvbuf = numpy.empty(sum(counts), dtype=mpi_dtype)

    sendbuf = sendbuf.ravel()
    for p0, p1 in lib.prange(0, numpy.max(counts), BLKSIZE):
        counts_seg = _segment_counts(counts, p0, p1)
        comm.Allgatherv([sendbuf[p0:p1], mpi_dtype],
                        [recvbuf, counts_seg, displs+p0, mpi_dtype])
    if split_recvbuf:
        return [recvbuf[p0:p0+c].reshape(shape)
                for p0,c,shape in zip(displs,counts,rshape)]
    else:
        try:
            return recvbuf.reshape((-1,) + shape[1:])
        except ValueError:
            return recvbuf

def alltoall(sendbuf, split_recvbuf=False):
    if isinstance(sendbuf, numpy.ndarray):
        mpi_dtype = comm.bcast(sendbuf.dtype.char)
        sendbuf = numpy.asarray(sendbuf, mpi_dtype, 'C')
        nrow = sendbuf.shape[0]
        ncol = sendbuf.size // nrow
        segsize = (nrow+mpi_size-1) // mpi_size * ncol
        sdispls = numpy.arange(0, mpi_size*segsize, segsize)
        sdispls[sdispls>sendbuf.size] = sendbuf.size
        scounts = numpy.append(sdispls[1:]-sdispls[:-1], sendbuf.size-sdispls[-1])
        rshape = comm.alltoall(scounts)
    else:
        _assert(len(sendbuf) == mpi_size)
        mpi_dtype = comm.bcast(sendbuf[0].dtype.char)
        sendbuf = [numpy.asarray(x, mpi_dtype) for x in sendbuf]
        rshape = comm.alltoall([x.shape for x in sendbuf])
        scounts = numpy.asarray([x.size for x in sendbuf])
        sdispls = numpy.append(0, numpy.cumsum(scounts[:-1]))
        sendbuf = numpy.hstack([x.ravel() for x in sendbuf])

    rcounts = numpy.asarray([numpy.prod(x) for x in rshape])
    rdispls = numpy.append(0, numpy.cumsum(rcounts[:-1]))
    recvbuf = numpy.empty(sum(rcounts), dtype=mpi_dtype)

    max_counts = max(numpy.max(scounts), numpy.max(rcounts))
    sendbuf = sendbuf.ravel()
    #DONOT use lib.prange. lib.prange may terminate early in some processes
    for p0, p1 in prange(0, max_counts, BLKSIZE):
        scounts_seg = _segment_counts(scounts, p0, p1)
        rcounts_seg = _segment_counts(rcounts, p0, p1)
        comm.Alltoallv([sendbuf, scounts_seg, sdispls+p0, mpi_dtype],
                       [recvbuf, rcounts_seg, rdispls+p0, mpi_dtype])

    if split_recvbuf:
        return [recvbuf[p0:p0+c].reshape(shape)
                for p0,c,shape in zip(rdispls,rcounts,rshape)]
    else:
        return recvbuf

def send(sendbuf, dest=0, tag=0):
    sendbuf = numpy.asarray(sendbuf, order='C')
    comm.send((sendbuf.shape, sendbuf.dtype), dest=dest, tag=tag)
    send_seg = numpy.ndarray(sendbuf.size, dtype=sendbuf.dtype, buffer=sendbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Send(send_seg[p0:p1], dest=dest, tag=tag)
    return sendbuf

def recv(source=0, tag=0):
    shape, dtype = comm.recv(source=source, tag=tag)
    recvbuf = numpy.empty(shape, dtype=dtype)
    recv_seg = numpy.ndarray(recvbuf.size, dtype=recvbuf.dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, recvbuf.size, BLKSIZE):
        comm.Recv(recv_seg[p0:p1], source=source, tag=tag)
    return recvbuf

def sendrecv(sendbuf, source=0, dest=0, tag=0):
    if source == dest:
        return sendbuf

    if rank == source:
        send(sendbuf, dest, tag)
    elif rank == dest:
        return recv(source, tag)

def rotate(sendbuf, blocking=True, tag=0):
    '''On every process, pass the sendbuf to the next process.
    Node-ID  Before-rotate  After-rotate
    node-0   buf-0          buf-1
    node-1   buf-1          buf-2
    node-2   buf-2          buf-3
    node-3   buf-3          buf-0
    '''
    if mpi_size <= 1:
        return sendbuf

    if rank == 0:
        prev_node = mpi_size - 1
        next_node = 1
    elif rank == mpi_size - 1:
        prev_node = rank - 1
        next_node = 0
    else:
        prev_node = rank - 1
        next_node = rank + 1

    if isinstance(sendbuf, numpy.ndarray):
        if blocking:
            if rank % 2 == 0:
                send(sendbuf, prev_node, tag)
                recvbuf = recv(next_node, tag)
            else:
                recvbuf = recv(next_node, tag)
                send(sendbuf, prev_node, tag)
        else:
            handler = lib.ThreadWithTraceBack(target=send, args=(sendbuf, prev_node, tag))
            handler.start()
            recvbuf = recv(next_node, tag)
            handler.join()
    else:
        if rank % 2 == 0:
            comm.send(sendbuf, dest=next_node, tag=tag)
            recvbuf = comm.recv(source=prev_node, tag=tag)
        else:
            recvbuf = comm.recv(source=prev_node, tag=tag)
            comm.send(sendbuf, dest=next_node, tag=tag)
    return recvbuf

def _assert(condition):
    if not condition:
        sys.stderr.write(''.join(traceback.format_stack()[:-1]))
        comm.Abort()

def _segment_counts(counts, p0, p1):
    counts_seg = counts - p0
    counts_seg[counts<=p0] = 0
    counts_seg[counts> p1] = p1 - p0
    return counts_seg

def prange(start, stop, step):
    '''Similar to lib.prange. This function ensures that all processes have the
    same number of steps.  It is required by alltoall communication.
    '''
    nsteps = (stop - start + step - 1) // step
    nsteps = max(comm.allgather(nsteps))
    for i in range(nsteps):
        i0 = min(stop, start + i * step)
        i1 = min(stop, i0 + step)
        yield i0, i1


n = 400
np = 500
n_seg = n // mpi_size
np_seg = np // mpi_size

def _rotate_tensor(buf, offsets):
    ntasks = mpi_size
    tasks = list(range(ntasks))
    tasks = tasks[rank:] + tasks[:rank]
    for task_id in tasks:
        if task_id != rank:
            buf = rotate(buf)
        col0 = offsets[task_id]
        col1 = offsets[task_id+1]
        yield task_id, buf, col0, col1 # row0, row1


if mpi_rank == mpi_size - 1:
    a = numpy.random.random((n_seg, n))
    b = numpy.random.random((n, np_seg))
else:
    a = numpy.random.random((n_seg, n))
    b = numpy.random.random((n, np_seg))

a_all = comm.gather(a)
b_all = comm.gather(b)
if mpi_rank == 0:
    a_all = numpy.vstack(a_all)
    b_all = numpy.hstack(b_all)

offsets = numpy.cumsum([0] + list(comm.allgather(n_seg)))

# a.dot(b)
c = numpy.zeros((n, b.shape[1]))
for task_id, a, col0, col1 in _rotate_tensor(a, offsets):
    c_priv = a.dot(b)
    c[col0:col1] = c_priv

c_all = comm.gather(c)
if mpi_rank == 0:
    c_all = numpy.hstack(c_all)
    print abs(a_all.dot(b_all) - c_all).max()
