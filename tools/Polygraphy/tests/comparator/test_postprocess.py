import numpy as np
from polygraphy.comparator import PostprocessFunc


class TestTopK(object):
    def test_basic(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        func = PostprocessFunc.topk_func(k=3)
        top_k = func({"x": arr})
        assert np.all(top_k["x"] == [4, 3, 2])


    def test_k_can_exceed_array_len(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        func = PostprocessFunc.topk_func(k=10)
        top_k = func({"x": arr})
        assert np.all(top_k["x"] == [4, 3, 2, 1, 0])
