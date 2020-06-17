#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Though omitted in this example, in some cases, it may be useful to embed
# shape information in the graph. We can use ONNX shape inference to do this:
#
# from onnx import shape_inference
# model = shape_inference.infer_shapes(onnx.load("model.onnx"))
#
# IMPORTANT: In some cases, ONNX shape inference may not correctly infer shapes,
# which will result in an invalid subgraph. To avoid this, you can instead modify
# the tensors to include the shape information yourself.

model = onnx.load("model.onnx")
graph = gs.import_onnx(model)

# Since we already know the names of the tensors we're interested in, we can
# grab them directly from the tensor map.
#
# NOTE: If you do not know the tensor names you want, you can view the graph in
# Netron to determine them, or use ONNX GraphSurgeon in an interactive shell
# to print the graph.
tensors = graph.tensors()

# If you want to embed shape information, but cannot use ONNX shape inference,
# you can manually modify the tensors at this point:
#
# graph.inputs = [tensors["x1"].to_variable(dtype=np.float32, shape=(1, 3, 224, 224))]
# graph.outputs = [tensors["add_out"].to_variable(dtype=np.float32, shape=(1, 3, 224, 224))]
#
# IMPORTANT: You must include type information for input and output tensors if it is not already
# present in the graph.
#
# NOTE: ONNX GraphSurgeon will also accept dynamic shapes - simply set the corresponding
# dimension(s) to `gs.Tensor.DYNAMIC`, e.g. `shape=(gs.Tensor.DYNAMIC, 3, 224, 224)`
graph.inputs = [tensors["x1"].to_variable(dtype=np.float32)]
graph.outputs = [tensors["add_out"].to_variable(dtype=np.float32)]

# Notice that we do not need to manually modify the rest of the graph. ONNX GraphSurgeon will
# take care of removing any unnecessary nodes or tensors, so that we are left with only the subgraph.
graph.cleanup()

onnx.save(gs.export_onnx(graph), "subgraph.onnx")
