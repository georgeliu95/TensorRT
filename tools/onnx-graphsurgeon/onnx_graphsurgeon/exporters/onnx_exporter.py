from onnx_graphsurgeon.exporters.base_exporter import BaseExporter
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Tensor, ConstantTensor, VariableTensor
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node

from collections import OrderedDict
from typing import Union
import onnx.numpy_helper
import numpy as np
import onnx


def dtype_to_onnx(dtype: np.dtype) -> int:
    return onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)]


class OnnxExporter(BaseExporter):
    @staticmethod
    def export_tensor_proto(tensor: ConstantTensor) -> onnx.TensorProto:
        onnx_tensor = onnx.numpy_helper.from_array(tensor.values)
        onnx_tensor.name = tensor.name
        return onnx_tensor


    @staticmethod
    def export_value_info_proto(tensor: Tensor) -> onnx.ValueInfoProto:
        if isinstance(tensor, ConstantTensor):
            onnx_tensor = onnx.helper.make_tensor_value_info(tensor.name, dtype_to_onnx(tensor.values.dtype), tensor.values.shape)
        elif isinstance(tensor, VariableTensor):
            onnx_tensor = onnx.helper.make_tensor_value_info(tensor.name, dtype_to_onnx(tensor.dtype), tensor.shape)
        elif isinstance(tensor, Tensor):
            onnx_tensor = onnx.helper.make_empty_tensor_value_info(tensor.name)
        return onnx_tensor


    @staticmethod
    def export_node(node: Node) -> onnx.NodeProto:
        # Cannot pass in attrs directly as make_node will change the order
        onnx_node = onnx.helper.make_node(node.op, inputs=[t.name for t in node.inputs], outputs=[t.name for t in node.outputs], name=node.name)
        # Convert Tensors and Graphs to TensorProtos and GraphProtos respectively
        for key, val in node.attrs.items():
            if isinstance(val, Tensor):
                val = OnnxExporter.export_tensor_proto(val)
            elif isinstance(val, Graph):
                val = OnnxExporter.export_graph(val)
            onnx_node.attribute.extend([onnx.helper.make_attribute(key, val)])
        return onnx_node


    @staticmethod
    def export_graph(graph: Graph) -> onnx.GraphProto:
        """
        Export an onnx-graphsurgeon Graph to an ONNX GraphProto.

        Args:
            graph (Graph): The graph to export.
        """
        nodes = [OnnxExporter.export_node(node) for node in graph.nodes]
        inputs = [OnnxExporter.export_value_info_proto(inp) for inp in graph.inputs]
        outputs = [OnnxExporter.export_value_info_proto(out) for out in graph.outputs]
        tensor_map = graph.generate_tensor_map()
        initializer = [OnnxExporter.export_tensor_proto(tensor) for tensor in tensor_map.values() if isinstance(tensor, ConstantTensor)]

        # Remove inputs and outputs to export ValueInfoProtos
        for tensor in graph.inputs + graph.outputs:
            if tensor.name in tensor_map:
                del tensor_map[tensor.name]

        value_info = [OnnxExporter.export_value_info_proto(tensor) for tensor in tensor_map.values()]
        return onnx.helper.make_graph(nodes=nodes, name=graph.name, inputs=inputs, outputs=outputs, initializer=initializer, doc_string=graph.doc_string, value_info=value_info)
