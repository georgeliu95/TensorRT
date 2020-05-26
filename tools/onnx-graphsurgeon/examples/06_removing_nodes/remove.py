#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import onnx

graph = gs.import_onnx(onnx.load("model.onnx"))

fake_node = [node for node in graph.nodes if node.op == "FakeNodeToRemove"][0]

# Get the input node of the fake node
# Node provides i() and o() functions that can optionally be provided an index (default is 0)
# These serve as convenience functions for the alternative, which would be to fetch the input/output
# tensor first, then fetch the input/output node of the tensor.
# For example, node.i() is equivalent to node.inputs[0].inputs[0]
inp_node = fake_node.i()

# Reconnect the input node to the output tensors of the fake node, so that the first identity
# node in the example graph now skips over the fake node.
inp_node.outputs = fake_node.outputs
fake_node.outputs.clear()

# Remove the fake node from the graph completely
graph.cleanup()
onnx.save(gs.export_onnx(graph), "removed.onnx")
