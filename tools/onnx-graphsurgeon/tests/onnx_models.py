from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Tensor, ConstantTensor, VariableTensor
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter

G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE

from collections import OrderedDict
import onnx.numpy_helper
from typing import List
import numpy as np
import onnx
import os

TEST_ROOT = os.path.realpath(os.path.dirname(__file__))

class Model(object):
    def __init__(self, path: str, inputs: List[Tensor], outputs: List[Tensor], nodes: List[Node], opset: int):
        self.path = path
        self.inputs = inputs
        self.outputs = outputs
        self.nodes = nodes
        self.opset = opset

    def load(self):
        return onnx.load(self.path)

    def assert_equal(self, graph: Graph):

        for actual, expected in zip(graph.inputs, self.inputs):
            assert actual == expected
        G_LOGGER.debug("Graph inputs matched")

        for actual, expected in zip(graph, self.nodes):
            G_LOGGER.debug("Actual Node: {:}.\n\nExpected Node: {:}".format(actual, expected))
            # Break down fields to make debugging failures easier.
            assert actual.op == expected.op
            assert actual.inputs == expected.inputs
            assert actual.outputs == expected.outputs
            assert actual.name == expected.name
            for (akey, aval), (ekey, eval) in zip(actual.attrs.items(), expected.attrs.items()):
                assert akey == ekey
                assert aval == eval
            assert actual == expected
        G_LOGGER.debug("Graph nodes matched")

        for actual, expected in zip(graph.outputs, self.outputs):
            assert actual == expected
        G_LOGGER.debug("Graph outputs matched")


def identity_model():
    path = os.path.join(TEST_ROOT, "models", "identity.onnx")
    model = onnx.load(path)

    x = VariableTensor(name="x", dtype=np.float32, shape=(1, 1, 2, 2))
    y = VariableTensor(name="y", dtype=np.float32, shape=(1, 1, 2, 2))
    node = Node(op="Identity", inputs=[x], outputs=[y])

    return Model(path, inputs=[x], outputs=[y], nodes=[node], opset=OnnxImporter.get_opset(model))


def dim_param_model():
    path = os.path.join(TEST_ROOT, "models", "dim_param.onnx")
    model = onnx.load(path)

    x = VariableTensor(name="Input:0", dtype=np.float32, shape=("dim0", 16, 128))
    y = VariableTensor(name="Output:0", dtype=np.float32, shape=("dim0", 16, 128))
    node = Node(op="Identity", inputs=[x], outputs=[y])

    return Model(path, inputs=[x], outputs=[y], nodes=[node], opset=OnnxImporter.get_opset(model))


def lstm_model():
    path = os.path.join(TEST_ROOT, "models", "lstm.onnx")
    model = onnx.load(path)
    onnx_graph = model.graph

    def load_initializer(index: int) -> np.ndarray:
        return onnx.numpy_helper.to_array(onnx_graph.initializer[index])

    # Optional inputs are represented by empty tensors
    X = VariableTensor(name="X", dtype=np.float32, shape=(4, 3, 6))
    W = ConstantTensor(name="W", values=load_initializer(0))
    R = ConstantTensor(name="R", values=load_initializer(1))
    B = ConstantTensor(name="B", values=load_initializer(2))
    initial_c = ConstantTensor(name="initial_c", values=load_initializer(3))

    Y = VariableTensor(name="Y", dtype=np.float32, shape=(4, 1, 3, 5))
    Y_h = VariableTensor(name="Y_h", dtype=np.float32, shape=(1, 3, 5))
    Y_c = VariableTensor(name="Y_c", dtype=np.float32, shape=(1, 3, 5))

    attrs = OrderedDict()
    attrs["direction"] = "forward"
    attrs["hidden_size"] = 5
    node = Node(op="LSTM", attrs=attrs, inputs=[X, W, R, B, Tensor.empty(), Tensor.empty(), initial_c], outputs=[Y, Y_h, Y_c])

    return Model(path, inputs=[X, W, R, B, initial_c], outputs=[Y, Y_h, Y_c], nodes=[node], opset=OnnxImporter.get_opset(model))


def scan_model():
    path = os.path.join(TEST_ROOT, "models", "scan.onnx")
    model = onnx.load(path)

    # Body graph
    sum_in = VariableTensor(name="sum_in", dtype=np.float32, shape=(2, ))
    next = VariableTensor(name="next", dtype=np.float32, shape=(2, ))
    sum_out = VariableTensor(name="sum_out", dtype=np.float32, shape=(2, ))
    scan_out = VariableTensor(name="scan_out", dtype=np.float32, shape=(2, ))

    body_nodes = [
        Node(op="Add", inputs=[sum_in, next], outputs=[sum_out]),
        Node(op="Identity", inputs=[sum_out], outputs=[scan_out]),
    ]
    body_graph = Graph(nodes=body_nodes, inputs=[sum_in, next], outputs=[sum_out, scan_out], name="scan_body")

    # Outer graph
    inputs = [
        VariableTensor(name="initial", dtype=np.float32, shape=(2, )),
        VariableTensor(name="x", dtype=np.float32, shape=(3, 2)),
    ]
    outputs = [
        VariableTensor(name="y", dtype=np.float32, shape=(2, )),
        VariableTensor(name="z", dtype=np.float32, shape=(3, 2)),
    ]

    attrs = OrderedDict()
    attrs["body"] = body_graph
    attrs["num_scan_inputs"] = 1
    scan_node = Node(op="Scan", inputs=inputs, outputs=outputs, attrs=attrs)
    return Model(path, inputs=inputs, outputs=outputs, nodes=[scan_node], opset=OnnxImporter.get_opset(model))