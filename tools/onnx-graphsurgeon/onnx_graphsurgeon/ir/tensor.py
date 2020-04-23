from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.util import misc

from typing import Set, Sequence, Union
import numpy as np


class Tensor(object):
    DYNAMIC = -1

    def __init__(self):
        raise NotImplementedError("Tensor is an abstract class")


    def __setattr__(self, name, value):
        if name in ["inputs", "outputs"]:
            try:
                getattr(self, name).clear()
                getattr(self, name).extend(value)
            except AttributeError:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


    def is_empty(self):
        """
        Returns whether this tensor is empty.

        Returns:
            bool: Whether the tensor is empty.
        """
        return self.name == ""


    def to_constant(self, values: np.ndarray):
        """
        Modifies this tensor in-place to convert it to a Constant. This means that all consumers/producers of the tensor will see the update.

        Args:
            values (np.ndarray): The values in this tensor

        Returns:
            self
        """
        self.__class__ = Constant
        self.values = values
        return self


    def to_variable(self, dtype: np.dtype=None, shape: Sequence[Union[int, str]]=[]):
        """
        Modifies this tensor in-place to convert it to a Variable. This means that all consumers/producers of the tensor will see the update.

        Optional Args:
            dtype (np.dtype): The data type of the tensor.
            shape (Sequence[int]): The shape of the tensor.

        Returns:
            self
        """
        self.__class__ = Variable
        self.dtype = dtype
        self.shape = shape
        return self


    def __str__(self):
        return "{:} ({:})".format(type(self).__name__, self.name)


    def __repr__(self):
        return self.__str__()


    def __eq__(self, other):
        """
        Perform a check to see if two tensors are equal.

        Tensors are considered equal if they share the same name. A Graph must not include Tensors with duplicate names.
        """
        return self.name == other.name


class Variable(Tensor):
    @staticmethod
    def empty():
        return Variable(name="")


    def __init__(self, name: str, dtype: np.dtype=None, shape: Sequence[Union[int, str]]=None):
        """
        Represents a Tensor whose value is not known until inference-time.

        Args:
            name (str): The name of the tensor.
            dtype (np.dtype): The data type of the tensor.
            shape (Sequence[Union[int, str]]): The shape of the tensor. This may contain strings if the model uses dimension parameters.
        """
        self.name = name
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=[])
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=[])
        self.dtype = dtype
        self.shape = misc.default_value(shape, None)


    def has_metadata(self):
        """
        Whether this tensor includes metadata about its data type and shape.

        Returns:
            bool
        """
        return self.dtype is not None and self.shape is not None


    def to_constant(self, values: np.ndarray):
        del self.dtype
        del self.shape
        return super().to_constant(values)


    def copy(self):
        """
        Makes a shallow copy of this tensor, omitting input and output information.

        Note: Generally, you should only ever make a deep copy of a Graph.
        """
        return Variable(self.name, self.dtype, self.shape)


    def __str__(self):
        return "{:}: (shape={:}, dtype={:})".format(super().__str__(), self.shape, self.dtype)


class Constant(Tensor):
    def __init__(self, name: str, values: np.ndarray):
        """
        Represents a Tensor whose value is known.

        Args:
            name (str): The name of the tensor.
            values (np.ndarray): The values in this tensor.
            dtype (np.dtype): The data type of the tensor.
            shape (Sequence[Union[int, str]]): The shape of the tensor.
        """
        self.name = name
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=[])
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=[])
        self.values = values


    def has_metadata(self):
        return True


    def to_variable(self, dtype: np.dtype=None, shape: Sequence[Union[int, str]]=[]):
        del self.values
        return super().to_variable(dtype, shape)


    def copy(self):
        """
        Makes a shallow copy of this tensor, omitting input and output information.

        Note: Generally, you should only ever make a deep copy of a Graph.
        """
        return Constant(self.name, self.values)


    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype.type

    def __repr__(self):
        ret = self.__str__()
        ret += "\n{:}".format(self.values)
        return ret
