from polygraphy.logger import G_LOGGER

from collections import OrderedDict

import numpy as np


class TensorMetadata(OrderedDict):
    """
    An OrderedDict[str, Tuple[np.dtype, Tuple[int]]] that maps input names to their data types and shapes.
    """
    def add(self, name, dtype, shape):
        """
        Convenience function for adding entries.

        Args:
            name (str): The name of the input.
            dtype (np.dtype): The data type of the input.
            shape (Tuple[int]):
                    The shape of the input. Dynamic dimensions may
                    be indicated by negative values, or ``None``.

        Returns:
            The newly added entry.
        """
        if shape is not None and any([not isinstance(dim, int) and dim is not None for dim in shape]):
            G_LOGGER.critical("Input Tensor: {:} | One or more elements in shape: {:} could not be understood. "
                              "Each shape value must be an integer or None".format(name, shape))

        self[name] = (dtype, shape)
        return self


    def __repr__(self):
        ret = "TensorMetadata()"
        for name, (dtype, shape) in self.items():
            ret += ".add('{:}', {:}, {:})".format(name, dtype, shape)
        return ret


    def __str__(self):
        sep = ", "
        elems = ["{:} [dtype={:}, shape={:}]".format(name, np.dtype(dtype).name, shape) for name, (dtype, shape) in self.items()]
        return sep.join(elems)
