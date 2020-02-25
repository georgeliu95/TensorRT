# ONNX GraphSurgeon changelog history

Dates are in YYYY-MM-DD format.

## v0.1.2 (2020-02-19)
- Adds `i()` and `o()` convenience functions to `Node` for retrieving input/output nodes.
- Adds `fold_constants()` to `Graph` to allow for folding constants in the graph.
- Adds `__deepcopy__()` to `Graph`.
- Adds `make_constant()` and `make_variable()` functions to `VariableTensor` and `ConstantTensor` respectively to transmute them in-place.

## v0.1.1 (2020-02-11)
- Removes some type annotations to allow compatibility with Python 3.5.

## v0.1.0 (2020-02-11)
- Adds `Node`, `Tensor` and `Graph` classes.
- Adds `BaseImporter` and `OnnxImporter` classes.
- Adds support for importing initializers in the `OnnxImporter`
- Adds `VariableTensor` and `ConstantTensor`
- Consolidates inputs/outputs of Nodes/Tensors. Now, inputs/outputs should generally only be added to `Node`s.
- Adds `OnnxExporter` to export `Graph` to `onnx.GraphProto`
- Adds `OnnxExporter` and `OnnxImporter` to public imports
- Adds `toposort` function to `Graph`, which will topologically sort it.
- Adds `cleanup` function to `Graph`, which will remove unused nodes and tensors.
- Adds high-level API for importing/exporting `Graph`s from/to ONNX models.
- `Graph`s are now generated with a default name of `onnx_graphsurgeon`
