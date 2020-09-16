import ctypes

import numpy as np
import pytest
from polygraphy.util.cuda import DeviceBuffer, Stream


class ResizeTestCase(object):
    # *_bytes is the size of the allocated buffer, old/new are the apparent shapes of the buffer.
    def __init__(self, old, old_size, new, new_size):
        self.old = old
        self.old_bytes = old_size * np.float32().itemsize
        self.new = new
        self.new_bytes = new_size * np.float32().itemsize

RESIZES = [
    ResizeTestCase(tuple(), 1, (1, 1, 1), 1), # Reshape (no-op)
    ResizeTestCase((2, 2, 2), 8, (1, 1), 8), # Resize to smaller buffer
    ResizeTestCase((2, 2, 2), 8, (9, 9), 81), # Resize to larger buffer
]

class TestDeviceBuffer(object):
    @pytest.mark.parametrize("shapes", RESIZES)
    def test_device_buffer_resize(self, shapes):
        buf = DeviceBuffer(shapes.old)
        assert buf.allocated_nbytes == shapes.old_bytes
        assert buf.shape == shapes.old
        buf.resize(shapes.new)
        assert buf.allocated_nbytes == shapes.new_bytes
        assert buf.shape == shapes.new


    def test_device_buffer_memcpy_async(self):
        stream = Stream()
        arr = np.ones((1, 384), dtype=np.int32)

        buf = DeviceBuffer()
        buf.resize(arr.shape)
        buf.copy_from(arr)

        new_arr = np.empty((1, 384), dtype=np.int32)
        buf.copy_to(new_arr, stream)

        stream.synchronize()

        assert np.all(new_arr == arr)


    def test_device_buffer_memcpy_sync(self):
        arr = np.ones((1, 384), dtype=np.int32)

        buf = DeviceBuffer()
        buf.resize(arr.shape)
        buf.copy_from(arr)

        new_arr = np.empty((1, 384), dtype=np.int32)
        buf.copy_to(new_arr)

        assert np.all(new_arr == arr)


class TestStream(object):
    def test_handle_is_ctypes_ptr(self):
        # If the handle is not a c_void_p (e.g. just an int), then it may cause segfaults when used.
        stream = Stream()
        assert isinstance(stream.handle, ctypes.c_void_p)
