from polygraphy.util.cuda import DeviceBuffer
import numpy as np

import pytest


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

@pytest.mark.parametrize("shapes", RESIZES)
def test_device_buffer_resize(shapes):
    buf = DeviceBuffer(shapes.old)
    assert buf.allocated_nbytes == shapes.old_bytes
    assert buf.shape == shapes.old
    buf.resize(shapes.new)
    assert buf.allocated_nbytes == shapes.new_bytes
    assert buf.shape == shapes.new
