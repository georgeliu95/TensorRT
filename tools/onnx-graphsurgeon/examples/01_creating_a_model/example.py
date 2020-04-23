#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import numpy as np
import onnx

X = gs.Variable(name="X", dtype=np.float32, shape=(1, 3, 5, 5))
Y = gs.Variable(name="Y", dtype=np.float32, shape=(1, 3, 1, 1))
node = gs.Node(op="GlobalLpPool", attrs={"p": 2}, inputs=[X], outputs=[Y])

graph = gs.Graph(nodes=[node], inputs=[X], outputs=[Y])
onnx.save(gs.export_onnx(graph), "test_globallppool.onnx")
