# ONNX GraphSurgeon


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
    - [Building From Source](#building-from-source)
        - [Using Make Targets](#using-make-targets)
        - [Building Manually](#building-manually)
- [Understanding The Basics](#understanding-the-basics)
    - [Importers](#importers)
    - [IR](#ir)
        - [Tensor](#tensor)
        - [Node](#node)
        - [A Note On Modifying Inputs And Outputs](#a-note-on-modifying-inputs-and-outputs)
        - [Graph](#graph)
    - [Exporters](#exporters)
- [Examples](#examples)
    - [Creating An ONNX Model By Hand](#creating-an-onnx-model-by-hand)
    - [Creating An ONNX Model With An Initializer](#creating-an-onnx-model-with-an-initializer)
    - [Isolating A Failing Node From A Model](#isolating-a-failing-node-from-a-model)
    - [Modifying A Graph In-Place](#modifying-a-graph-in-place)

## Introduction

ONNX GraphSurgeon is a tool that allows you to easily generate new ONNX graphs, or modify existing ones.


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


## Understanding The Basics

ONNX GraphSurgeon is composed of three major components: Importers, the IR, and Exporters.

### Importers

Importers are used to import a graph into the ONNX GraphSurgeon IR.
The importer interface is defined in [base_importer.py](./onnx_graphsurgeon/importers/base_importer.py).

ONNX GraphSurgeon also provides [high-level importer APIs](./onnx_graphsurgeon/api/api.py) for ease of use.

### IR

The Intermediate Representation (IR) is where all modifications to the graph are made. It can also be used to create new graphs from scratch.
The IR involves three components: [Tensor](./onnx_graphsurgeon/ir/tensor.py)s, [Node](./onnx_graphsurgeon/ir/node.py)s, and [Graph](./onnx_graphsurgeon/ir/graph.py)s.
Nearly all of the members of each component can be freely modified.

#### Tensor

Tensors are divided into two subclasses: `VariableTensor` and `ConstantTensor`.

A `ConstantTensor` is a tensor whose values are known upfront, and can be retrieved as a NumPy array and modified, whereas the values of a `VariableTensor` are unknown until inference-time.

The inputs and outputs of Tensors are always Nodes.

**An example constant tensor from ResNet50:**
```
>>> print(tensor)
ConstantTensor (gpu_0/res_conv1_bn_s_0)
[0.85369843 1.1515082  0.9152944  0.9577646  1.0663182  0.55629414
 1.2009839  1.1912311  2.2619808  0.62263143 1.1149117  1.4921428
 0.89566356 1.0358194  1.431092   1.5360111  1.25086    0.8706703
 1.2564877  0.8524589  0.9436758  0.7507614  0.8945271  0.93587166
 1.8422242  3.0609846  1.3124607  1.2158023  1.3937513  0.7857263
 0.8928106  1.3042281  1.0153942  0.89356416 1.0052011  1.2964457
 1.1117343  1.0669073  0.91343874 0.92906713 1.0465593  1.1261675
 1.4551278  1.8252873  1.9678202  1.1031747  2.3236883  0.8831993
 1.1133649  1.1654979  1.2705412  2.5578163  0.9504889  1.0441847
 1.0620039  0.92997414 1.2119316  1.3101407  0.7091761  0.99814713
 1.3404484  0.96389204 1.3435135  0.9236031 ]
```

**An example variable tensor from ResNet50:**
```
>>> print(tensor)
VariableTensor (gpu_0/data_0): (shape=[1, 3, 224, 224], dtype=float32)
```


#### Node

A `Node` defines an operation in the graph, and can have zero or more attributes. Attribute values can be any Python primitive types, as well as ONNX GraphSurgeon `Graph`s or `Tensor`s

The inputs and outputs of Nodes are always Tensors

**An example ReLU node from ResNet50:**
```
>>> print(node)
 (Relu)
    Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
Attributes: OrderedDict()
```


#### A Note On Modifying Inputs And Outputs

The `inputs`/`outputs` members of nodes and tensors have special logic that will update the inputs/outputs of all affected nodes/tensors when you make a change.
This means, for example, that you do **not** need to update the `inputs` of a Node when you make a change to the `outputs` of its input tensor.

Consider the following node:
```
>>> print(node)
 (Relu).
    Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
Attributes: OrderedDict()
```

The input tensor can be accessed like so:
```
>>> tensor = node.inputs[0]
>>> print(tensor)
Tensor (gpu_0/res_conv1_bn_1)
>>> print(tensor.outputs)
[ (Relu).
	Inputs: [Tensor (gpu_0/res_conv1_bn_1)]
	Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
Attributes: OrderedDict()]
```

If we remove the node from the outputs of the tensor, this is reflected in the node inputs as well:
```
>>> del tensor.outputs[0]
>>> print(tensor.outputs)
[]
>>> print(node)
 (Relu).
    Inputs: []
    Outputs: [Tensor (gpu_0/res_conv1_bn_2)]
Attributes: OrderedDict()
```


#### Graph

A `Graph` contains zero or more `Node`s and input/output `Tensor`s.

Intermediate tensors are not explicitly tracked, but are instead retrieved from the nodes contained within the graph.


### Exporters

Exporters are used to export the ONNX GraphSurgeon IR to ONNX or other types of graphs.
The exporter interface is defined in [base_exporter.py](./onnx_graphsurgeon/exporters/base_exporter.py).

ONNX GraphSurgeon also provides [high-level exporter APIs](./onnx_graphsurgeon/api/api.py) for ease of use.


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

### Creating An ONNX Model With An Initializer

The following code creates an ONNX model containing a single Convolution node, with weights:
```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

inp = gs.VariableTensor(name="X", dtype=np.float32, shape=(1, 3, 224, 224))
# Since filter is a ConstantTensor, it will automatically be exported as an initializer
filter = gs.ConstantTensor(name="W", values=np.ones(shape=(5, 3, 3, 3), dtype=np.float32))

out = gs.VariableTensor(name="Y", dtype=np.float32, shape=(1, 5, 222, 222))

node = gs.Node(op="Conv", inputs=[inp, filter], outputs=[out])

# Note that initializers do not necessarily have to be graph inputs
graph = gs.Graph(nodes=[node], inputs=[inp], outputs=[out])
onnx.save(gs.export_onnx(graph), "test_conv.onnx")
```


### Isolating A Failing Node From A Model

Assume that `model.onnx` is an ONNX model where a node named `failing_node` is failing.

To figure out why, we can isolate the node into a separate ONNX graph, and use that as a unit test:
```python
import onnx_graphsurgeon as gs
import numpy as np
import onnx

graph = gs.import_onnx(onnx.load("model.onnx"))

# Isolate the failing node, then detach it from the graph by swapping out the inputs and outputs we care about.
failing_node = [node for node in graph if node.name == "failing_node"][0]
# ONNX requires I/O tensors to have dtype and shape information. Here we hard-code it,
# but it can also be derived automatically by running shape inference over the graph prior to importing.
# In that case, we would not need to replace the inputs and outputs of the node.
# See onnx.shape_inference.infer_shapes().
failing_node.inputs[0] = gs.VariableTensor(name="input", dtype=np.float32, shape=(-1, 128, 14, 14))
failing_node.outputs[0] = gs.VariableTensor(name="output", dtype=np.float32, shape=(-1, 128, 28, 28))

new_graph = gs.Graph(nodes=[failing_node], inputs=failing_node.inputs, outputs=failing_node.outputs)

onnx.save(gs.export_onnx(new_graph), "failing.onnx")
```

This will generate a new ONNX model called `failing.onnx` containing the
failing node, as well as any parameters or initializers (e.g. weights) associated
with the node (these do **not** need to be copied into the new graph manually!)


### Modifying A Graph In-Place

Assume that `model.onnx` is an ONNX model using an old opset containing `ATen` ops which are used to
perform a `Gather` operation.

We can modify the graph to replace the nodes and remove any extra inputs they might have:
```python
import onnx_graphsurgeon as gs
import onnx

graph = gs.import_onnx(onnx.load("model.onnx"))

aten_nodes = [node for node in graph if node.op == "ATen" and node.attrs["operator"] == "embedding_bag"]
for node in aten_nodes:
   node.op = "Gather"
   node.inputs = node.inputs[0:2]
   node.attrs = {"axis": 0}

onnx.save(gs.export_onnx(graph.cleanup()), "model_with_gather.onnx")
```
