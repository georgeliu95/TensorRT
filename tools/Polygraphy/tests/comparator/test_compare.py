import numpy as np
from polygraphy.comparator import CompareFunc, IterationResult


class TestCompareFunc(object):
    def test_basic_compare_func_can_compare_bool(self):
        iter_result0 = IterationResult(outputs={"output": np.zeros((4, 4), dtype=np.bool)})
        iter_result1 = IterationResult(outputs={"output": np.ones((4, 4), dtype=np.bool)})

        compare_func = CompareFunc.basic_compare_func()
        compare_func(iter_result0, iter_result1)
