#!/usr/bin/env python3
import onnx_graphsurgeon as gs
import onnx

graph = gs.import_onnx(onnx.load("model.onnx"))

# 1. Remove the `b` input of the add node
first_add = [node for node in graph.nodes if node.op == "Add"][0]
first_add.inputs = [inp for inp in first_add.inputs if inp.name != "b"]

# 2. Change the Add to a LeakyRelu
first_add.op = "LeakyRelu"
first_add.attrs["alpha"] = 0.02

# 3. Add an identity after the add node
identity_out = gs.Variable("identity_out")
identity = gs.Node(op="Identity", inputs=first_add.outputs, outputs=[identity_out])
graph.nodes.append(identity)

# 4. Modify the graph output to be the identity output
graph.outputs = [identity_out]

# 5. Remove unused nodes/tensors, and topologically sort the graph
# ONNX requires nodes to be topologically sorted to be considered valid.
# Therefore, you should only need to sort the graph when you have added new nodes out-of-order.
# In this case, the identity node is already in the correct spot (it is the last node,
# and was appended to the end of the list), but to be on the safer side, we can sort anyway.
graph.cleanup().toposort()

onnx.save(gs.export_onnx(graph), "modified.onnx")
