from onnx_graphsurgeon.exporters.base_exporter import BaseExporter
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Tensor, Constant, Variable
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
    def export_tensor_proto(tensor: Constant) -> onnx.TensorProto:
        onnx_tensor = onnx.numpy_helper.from_array(tensor.values)
        onnx_tensor.name = tensor.name
        return onnx_tensor


    @staticmethod
    def export_value_info_proto(tensor: Tensor, do_type_check: bool) -> onnx.ValueInfoProto:
        if isinstance(tensor, Constant):
            onnx_tensor = onnx.helper.make_tensor_value_info(tensor.name, dtype_to_onnx(tensor.values.dtype), tensor.values.shape)
        elif isinstance(tensor, Variable):
            if do_type_check and tensor.dtype is None:
                G_LOGGER.critical("Graph input and output tensors must include dtype information. Please set the dtype attribute for: {:}".format(tensor))

            if tensor.dtype is not None:
                onnx_tensor = onnx.helper.make_tensor_value_info(tensor.name, dtype_to_onnx(tensor.dtype), tensor.shape)
            else:
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
    def export_graph(graph: Graph, do_type_check=True) -> onnx.GraphProto:
        """
        Export an onnx-graphsurgeon Graph to an ONNX GraphProto.

        Args:
            graph (Graph): The graph to export.

        Optional Args:
            do_type_check (bool): Whether to check that input and output tensors have data types defined, and fail if not.
        """
        nodes = [OnnxExporter.export_node(node) for node in graph.nodes]
        inputs = [OnnxExporter.export_value_info_proto(inp, do_type_check) for inp in graph.inputs]
        outputs = [OnnxExporter.export_value_info_proto(out, do_type_check) for out in graph.outputs]
        tensor_map = graph.tensors()
        initializer = [OnnxExporter.export_tensor_proto(tensor) for tensor in tensor_map.values() if isinstance(tensor, Constant)]

        # Remove inputs and outputs to export ValueInfoProtos
        for tensor in graph.inputs + graph.outputs:
            if tensor.name in tensor_map:
                del tensor_map[tensor.name]

        # Omit tensors if we don't know their shape/type
        value_info = [OnnxExporter.export_value_info_proto(tensor, do_type_check) for tensor in tensor_map.values() if isinstance(tensor, Variable) and tensor.dtype is not None]
        return onnx.helper.make_graph(nodes=nodes, name=graph.name, inputs=inputs, outputs=outputs, initializer=initializer, doc_string=graph.doc_string, value_info=value_info)
