# ONNX GraphSurgeon


## Installation

### Building From Source

#### Using Make Targets
```
make install
```
Or, if installing inside a virtual environment:
```
make install_venv
```

#### Building Manually

1. Build a wheel:
```
make build
```

2. Install the wheel manually from **outside** the repository:
```
python3 -m pip install onnx_graphsurgeon/dist/onnx_graphsurgeon-X.Y.Z-py2.py3-none-any.whl --user
```
where `X.Y.Z` is the version number.


## Examples

### Creating An ONNX Model By Hand

The following code creates an ONNX model containing a single GlobalLpPool node:
```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

inp = gs.VariableTensor(name="X", dtype=np.float32, shape=(1, 3, 5, 5))
out = gs.VariableTensor(name="Y", dtype=np.float32, shape=(1, 3, 1, 1))
node = gs.Node(op="GlobalLpPool", attrs={"p": 2}, inputs=[inp], outputs=[out])

graph = gs.Graph(nodes=[node], inputs=[inp], outputs=[out])
onnx.save(gs.export_onnx(graph), "test_globallppool.onnx")
```

### Isolating A Failing Node From A Model

Assume that `model.onnx` is some ONNX model where a node named `failing_node` is failing.

To figure out why, we can isolate the node into a separate ONNX graph, and use that as a unit test.

```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

graph = gs.import_onnx(onnx.load("model.onnx"))

# Isolate the failing node, then detach it from the graph by swapping out the inputs and outputs we care about.
failing_node = [node for node in graph if node.name == "failing_node"][0]
# ONNX requires I/O tensors to have dtype and shape information. Here we hard-code it,
# but it can also be derived automatically by running shape inference over the graph prior to importing.
# See onnx.shape_inference.infer_shapes().
failing_node.inputs[0] = gs.VariableTensor(name="input", dtype=np.float32, shape=(-1, 128, 14, 14))
failing_node.outputs[0] = gs.VariableTensor(name="output", dtype=np.float32, shape=(-1, 128, 28, 28))

new_graph = gs.Graph(nodes=[failing_node], inputs=failing_node.inputs, outputs=failing_node.outputs)

onnx.save(gs.export_onnx(new_graph), "failing.onnx")
```

This will generate a new ONNX model called `failing.onnx` containing the
failing node, as well as any parameters or initializers (e.g. weights) associated
with the node (these do **not** need to be copied into the new graph manually!)
