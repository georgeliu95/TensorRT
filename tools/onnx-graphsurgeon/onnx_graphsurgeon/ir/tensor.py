from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.utils import misc

from typing import Set, Sequence
import numpy as np

class Tensor(object):
    DYNAMIC = -1

    def __init__(self, name: str):
        """
        A tensor is produced by at most one Node, and can be consumed by zero or more nodes.

        Args:
            name (str): The name of the tensor. Tensor names must be unique per graph, and must not be modified after being set.
        """
        self.name = name
        self.inputs = []
        self.outputs = []

    @staticmethod
    def empty():
        """
        Creates an empty tensor.
        """
        return Tensor("")

    def is_empty(self):
        return not self.name

    def __str__(self):
        return "{:} ({:})".format(type(self).__name__, self.name)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        """
        Perform a check to see if two tensors are equal.
        """
        return self.name == other.name


class VariableTensor(Tensor):
    def __init__(self, name: str, dtype: np.dtype, shape: Sequence[int]=[]):
        """
        Represents a Tensor whose value is not known until inference-time.

        Args:
            name (str): The name of the tensor. Tensor names must be unique per graph, and must not be modified after being set.
            dtype (np.dtype): The data type of the tensor.
            shape (Sequence[int]): The shape of the tensor.
        """
        super().__init__(name)
        self.dtype = dtype
        self.shape = list(misc.default_value(shape, []))


    def __str__(self):
        return "{:}: (shape={:}, dtype={:})".format(super().__str__(), self.shape, self.dtype)


class ConstantTensor(Tensor):
    def __init__(self, name: str, values: np.ndarray):
        """
        Represents a Tensor whose value is known.

        Args:
            name (str): The name of the tensor. Tensor names must be unique per graph, and must not be modified after being set.
            values (np.ndarray): The values in this tensor.
        """
        super().__init__(name)
        self.values = values

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def __repr__(self):
        ret = self.__str__()
        ret += "\tContents:\n{:}".format(self.values)
        return ret
