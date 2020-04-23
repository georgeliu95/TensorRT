#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import onnx

graph = gs.import_onnx(onnx.load("model.onnx"))

# Fold constants in the graph using ONNX Runtime. This will replace
# expressions that can be evaluated prior to runtime with constant tensors.
# The `fold_constants()` function will not, however, remove the nodes that
# it replaced - it simply changes the inputs of subsequent nodes.
# To remove these unused nodes, we can follow up `fold_constants()` with `cleanup()`
graph.fold_constants().cleanup()

onnx.save(gs.export_onnx(graph), "folded.onnx")
