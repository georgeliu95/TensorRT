from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Tensor
from onnx_graphsurgeon.utils import misc

from collections import OrderedDict
from typing import List, Dict

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
        self.inputs = misc.SynchronizedList(self, field_name="outputs", initial=misc.default_value(inputs, []))
        self.outputs = misc.SynchronizedList(self, field_name="inputs", initial=misc.default_value(outputs, []))


    def i(self, tensor_idx=0, producer_idx=0):
        """
        Convenience function to get a producer node of one of this node's inputs.

        Args:
            tensor_idx (int): The index of the input tensor of this node. Defaults to 0.
            producer_idx (int): The index of the producer of the input tensor, if the tensor has multiple producers. Defaults to 0

        Returns:
            Node: The specified producer (input) node.
        """
        return self.inputs[tensor_idx].inputs[producer_idx]


    def o(self, consumer_idx=0, tensor_idx=0):
        """
        Convenience function to get a consumer node of one of this node's outputs.

        Args:
            consumer_idx (int): The index of the consumer of the input tensor. Defaults to 0.
            tensor_idx (int): The index of the output tensor of this node, if the node has multiple outputs. Defaults to 0.
        """
        return self.outputs[tensor_idx].outputs[consumer_idx]


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
