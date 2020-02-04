from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Tensor
from onnx_graphsurgeon.utils import misc

from collections import OrderedDict
from typing import List, Dict, Sequence

# Special type of list that appends a node to the provided field of any tensor that is appended to the
# list (and correspondingly removes nodes) from removed tensors
class NodeIOList(list):
    def __init__(self, node, tensor_field, initial_list):
        self.node = node
        self.tensor_field = tensor_field
        self.extend(initial_list)


    def _add_to_tensor(self, tensor: Tensor):
        getattr(tensor, self.tensor_field).append(self.node)


    def _remove_from_tensor(self, tensor: Tensor):
        getattr(tensor, self.tensor_field).remove(self.node)


    def __setitem__(self, index, tensor: Tensor):
        self._remove_from_tensor(self[index])
        super().__setitem__(index, tensor)
        self._add_to_tensor(tensor)


    def append(self, x: Tensor):
        super().append(x)
        self._add_to_tensor(x)


    def extend(self, iterable: Sequence[Tensor]):
        super().extend(iterable)
        for tensor in iterable:
            self._add_to_tensor(tensor)

    def insert(self, i, x):
        super().insert(i, x)
        self._add_to_tensor(x)


    def remove(self, x):
        super().remove(x)
        self._remove_from_tensor(x)


    def pop(self, i=-1):
        tensor = super().pop(i)
        self._remove_from_tensor(tensor)
        return tensor


    def clear(self):
        for tensor in self:
            self._remove_from_tensor(tensor)
        super().clear()


    def __add__(self, other_list: List[Tensor]):
        new_list = NodeIOList(self.node, self.tensor_field, self)
        new_list += other_list
        return new_list


    def __iadd__(self, other_list: List[Tensor]):
        self.extend(other_list)
        return self


class Node(object):
    def __init__(self, op: str, name: str=None, attrs: Dict[str, object]=None, inputs: List["Tensor"]=None, outputs: List["Tensor"]=None):
        """
        A node consumes zero or more Tensors, and produces zero or more Tensors.

        Args:
            op (str): The operation this node performs.


        Optional Args:
            name (str): The name of this node.
            attrs (Dict[str, object]): A dictionary that maps attribute names to their values.
            inputs (List[Tensor]): A list of zero or more input Tensors.
            outputs (List[Tensor]): A list of zero or more output Tensors.
        """
        self.op = op
        self.name = misc.default_value(name, "")
        self.attrs = misc.default_value(attrs, OrderedDict())
        self.inputs = NodeIOList(self, tensor_field="outputs", initial_list=misc.default_value(inputs, []))
        self.outputs = NodeIOList(self, tensor_field="inputs", initial_list=misc.default_value(outputs, []))


    def __setattr__(self, name, value):
        if name in ["inputs", "outputs"] and hasattr(self, name):
            getattr(self, name).clear()
            getattr(self, name).extend(value)
        else:
            super().__setattr__(name, value)


    def __str__(self):
        return "{:} ({:}).\n\tInputs: {:}\n\tOutputs: {:}\nAttributes: {:}".format(self.name, self.op, self.inputs, self.outputs, self.attrs)


    def __repr__(self):
        return self.__str__()


    def __eq__(self, other):
        """
        Check whether two nodes are equal by comparing name, attributes, op, inputs, and outputs.
        """
        G_LOGGER.verbose("Comparing node: {:} with {:}".format(self.name, other.name))
        attrs_match = self.name == other.name and self.op == other.op and self.attrs == other.attrs
        inputs_match = len(self.inputs) == len(other.inputs) and all([inp == other_inp for inp, other_inp in zip(self.inputs, other.inputs)])
        outputs_match = len(self.outputs) == len(other.outputs) and all([out == other_out for out, other_out in zip(self.outputs, other.outputs)])
        return attrs_match and inputs_match and outputs_match
