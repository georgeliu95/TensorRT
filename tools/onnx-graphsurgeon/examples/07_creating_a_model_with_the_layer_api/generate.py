#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import numpy as np
import onnx

print("Graph.layer Help:\n{}".format(gs.Graph.layer.__doc__))

# We can use `Graph.register` to add a function to the Graph class. Later, we can invoke the function
# directly on instances of the graph, e.g., `graph.add(...)`
@gs.Graph.register
def add(self, a, b):
    # The Graph.layer function creates a node, adds inputs and outputs to it, and finally adds it to the graph.
    # It returns the output tensors of the node to make it easy to chain.
    # The function will append an index to any strings provided for inputs/outputs prior
    # to using them to construct tensors. This will ensure that multiple calls to the layer() function
    # will generate distinct tensors. However, this does NOT guarantee that there will be no overlap with
    # other tensors in the graph. Hence, you should choose the prefixes to minimize the possibility of
    # collisions.
    return self.layer(op="Add", inputs=[a, b], outputs=["add_out_gs"])


@gs.Graph.register
def mul(self, a, b):
    return self.layer(op="Mul", inputs=[a, b], outputs=["mul_out_gs"])


@gs.Graph.register
def gemm(self, a, b, trans_a=False, trans_b=False):
    attrs = {"transA": int(trans_a), "transB": int(trans_b)}
    return self.layer(op="Gemm", inputs=[a, b], outputs=["gemm_out_gs"], attrs=attrs)


@gs.Graph.register
def relu(self, a):
    return self.layer(op="Relu", inputs=[a], outputs=["act_out_gs"])


##########################################################################################################
# The functions registered above greatly simplify the process of building the graph itself.

graph = gs.Graph()

# Generates a graph which computes:
# output = ReLU((A * X^T) + B) (.) C + D
X = gs.Variable(name="X", shape=(64, 64), dtype=np.float32)
graph.inputs = [X]

# axt = (A * X^T)
# Note that we can use NumPy arrays directly (e.g. Tensor A),
# instead of Constants. These will automatically be converted to Constants.
A = np.ones(shape=(64, 64), dtype=np.float32)
axt = graph.gemm(A, X, trans_b=True)

# dense = ReLU(axt + B)
B = np.ones((64, 64), dtype=np.float32) * 0.5
dense = graph.relu(*graph.add(*axt, B))

# output = dense (.) C + D
# If a Tensor instance is provided (e.g. Tensor C), it will not be modified at all.
# If you prefer to set the exact names of tensors in the graph, you should
# construct tensors manually instead of passing strings or NumPy arrays.
C = gs.Constant(name="C", values=np.ones(shape=(64, 64), dtype=np.float32))
D = np.ones(shape=(64, 64), dtype=np.float32)
graph.outputs = graph.add(*graph.mul(*dense, C), D)
graph.outputs[0].dtype = np.float32

onnx.save(gs.export_onnx(graph), "model.onnx")
