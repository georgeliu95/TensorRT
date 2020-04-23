#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import numpy as np
import onnx

X = gs.Variable(name="X", dtype=np.float32, shape=(1, 3, 224, 224))
# Since W is a Constant, it will automatically be exported as an initializer
W = gs.Constant(name="W", values=np.ones(shape=(5, 3, 3, 3), dtype=np.float32))

Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 5, 222, 222))

node = gs.Node(op="Conv", inputs=[X, W], outputs=[Y])

# Note that initializers do not necessarily have to be graph inputs
graph = gs.Graph(nodes=[node], inputs=[X], outputs=[Y])
onnx.save(gs.export_onnx(graph), "test_conv.onnx")
