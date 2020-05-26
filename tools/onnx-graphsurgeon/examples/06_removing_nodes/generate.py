#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import numpy as np
import onnx

# Inputs
x = gs.Variable(name="x", dtype=np.float32, shape=(1, 3, 224, 224))

# Intermediate tensors
i0 = gs.Variable(name="i0")
i1 = gs.Variable(name="i1")

# Outputs
y = gs.Variable(name="y")

nodes = [
    gs.Node(op="Identity", inputs=[x], outputs=[i0]),
    gs.Node(op="FakeNodeToRemove", inputs=[i0], outputs=[i1]),
    gs.Node(op="Identity", inputs=[i1], outputs=[y]),
]

graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[y])
onnx.save(gs.export_onnx(graph), "model.onnx")
