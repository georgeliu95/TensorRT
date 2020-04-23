#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Computes outputs = input + ((a + b) + d)

shape = (1, 3)
# Inputs
input = gs.Variable("input", shape=shape, dtype=np.float32)

# Intermediate tensors
a = gs.Constant("a", values=np.ones(shape=shape, dtype=np.float32))
b = gs.Constant("b", values=np.ones(shape=shape, dtype=np.float32))
c = gs.Variable("c")
d = gs.Constant("d", values=np.ones(shape=shape, dtype=np.float32))
e = gs.Variable("e")

# Outputs
output = gs.Variable("output", shape=shape, dtype=np.float32)

nodes = [
    # c = (a + b)
    gs.Node("Add", inputs=[a, b], outputs=[c]),
    # e = (c + d)
    gs.Node("Add", inputs=[c, d], outputs=[e]),
    # output = input + e
    gs.Node("Add", inputs=[input, e], outputs=[output]),
]

graph = gs.Graph(nodes=nodes, inputs=[input], outputs=[output])
onnx.save(gs.export_onnx(graph), "model.onnx")
