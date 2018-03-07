from __future__ import absolute_import

from ._base import _LIB, check_call, c_array
from . import ndarray as _nd

import ctypes
import numpy as np


class DLContext(ctypes.Structure):
    _fields_ = [("device_id", ctypes.c_int),
                ("device_type", ctypes.c_int)]

    MASK2STR = {
        1: 'cpu',
        2: 'gpu',
    }

    def __init__(self, device_id, device_type):
        super(DLContext, self).__init__()
        self.device_id = device_id
        self.device_type = device_type

    def __repr__(self):
        return "%s(%d)" % (DLContext.MASK2STR[self.device_type], self.device_id)


class DLArray(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("ctx", DLContext),
                ("ndim", ctypes.c_int),
                ("shape", ctypes.POINTER(ctypes.c_int64))]


# 指针类型
DLArrayHandle = ctypes.POINTER(DLArray)


def cpu(dev_id=0):
    return DLContext(dev_id, 1)


def gpu(dev_id=0):
    return DLContext(dev_id, 2)


def is_gpu_ctx(ctx):
    return ctx and ctx.device_type == 2


class NDArray(object):
    __slots__ = ['handle']

    def __init__(self, handle):
        self.handle = handle

    def __del__(self):
        check_call(_LIB.DLArrayFree(self.handle))

    @property
    def shape(self):
        return tuple(self.handle.contents.shape[i]
                     for i in range(self.handle.contents.ndim))

    @property
    def ctx(self):
        return self.handle.contents.ctx

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                    in_slice.start is not None
            or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArray):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self._sync_copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def _sync_copyfrom(self, source_array):
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=np.float32)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported'
                                % str(type(source_array)))
        source_array = np.ascontiguousarray(source_array, dtype=np.float32)
        if source_array.shape != self.shape:
            raise ValueError('array shape do not match the shape of NDArray')
        source_arr, shape = NDArray._numpyasarray(source_array)
        check_call(_LIB.DLArrayCopyFromTo(
            ctypes.byref(source_arr), self.handle, None))
        # de-allocate shape until now
        _ = shape

    @staticmethod
    def _numpyasarray(np_data):
        data = np_data
        assert data.flags['C_CONTIGUOUS']
        arr = DLArray()
        shape = c_array(ctypes.c_int64, data.shape)
        arr.data = data.ctypes.data_as(ctypes.c_void_p)
        arr.shape = shape
        arr.ndim = data.ndim
        arr.ctx = cpu(0)
        return arr, shape

    def asnumpy(self):
        np_arr = np.empty(self.shape, dtype=np.float32)
        arr, shape = NDArray._numpyasarray(np_arr)
        check_call(_LIB.DLArrayCopyFromTo(
            self.handle, ctypes.byref(arr), None))
        _ = shape
        return np_arr

    def copyto(self, target):
        if isinstance(target, DLContext):
            target = empty(self.shape, target)
        if isinstance(target, NDArray):
            check_call(_LIB.DLArrayCopyFromTo(
                self.handle, target.handle, None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target


def array(arr, ctx=cpu(0)):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    ret = empty(arr.shape, ctx)
    ret._sync_copyfrom(arr)
    return ret


def empty(shape, ctx=cpu(0)):
    shape = c_array(ctypes.c_int64, shape)
    ndim = ctypes.c_int(len(shape))
    handle = DLArrayHandle()
    check_call(_LIB.DLArrayAlloc(
        shape, ndim, ctx, ctypes.byref(handle)))
    return NDArray(handle)


def reshape(arr, new_shape):
    assert isinstance(arr, _nd.NDArray)
    shape = c_array(ctypes.c_int64, new_shape)
    new_dim = len(new_shape)
    handle = arr.handle
    check_call(_LIB.DLArrayReshape(handle, shape, new_dim))
