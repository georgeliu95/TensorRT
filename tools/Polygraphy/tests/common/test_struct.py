import numpy as np
from polygraphy.common import TensorMetadata


class TestTensorMetadata(object):
    def test_str(self):
        meta = TensorMetadata().add("X", dtype=np.float32, shape=(64, 64))
        assert str(meta) == "{X [dtype=float32, shape=(64, 64)]}"


    def test_str_no_dtype(self):
        meta = TensorMetadata().add("X", dtype=None, shape=(64, 64))
        assert str(meta) == "{X [shape=(64, 64)]}"


    def test_str_no_shape(self):
        meta = TensorMetadata().add("X", dtype=np.float32, shape=None)
        assert str(meta) == "{X [dtype=float32]}"


    def test_str_no_meta(self):
        meta = TensorMetadata().add("X", dtype=None, shape=None)
        assert str(meta) == "{X}"
