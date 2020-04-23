# ONNX GraphSurgeon changelog history

Dates are in YYYY-MM-DD format.

## v0.2.0 (2020-04-15)
- Various improvements to the logger
- Adds an `examples` directory
- Updates `OnnxImporter` so that it can correctly import shapes and types from an ONNX graph after shape inference.
- Makes `Tensor` an abstract class - all tensors in a graph are now either `Variable` or `Constant`
- Adds `has_metadata()` to `Tensor` classes to determine if dtype/shape are known.
- Removes `Tensor` suffix from Tensor classes.
- Renames `generate_tensor_map()` to `tensors()` in `Graph`
- Adds a `check_duplicates` parameter to `Graph.tensors()` to make it easy to check for duplicate tensors in the graph.

## v0.1.3 (2020-02-26)
- The `import_onnx` and `export_onnx` functions will now preserve opset information and `dim_param` values in shapes.

## v0.1.2 (2020-02-19)
- Adds `i()` and `o()` convenience functions to `Node` for retrieving input/output nodes.
- Adds `fold_constants()` to `Graph` to allow for folding constants in the graph.
- Adds `__deepcopy__()` to `Graph`.
- Adds `to_constant()` and `to_variable()` functions to `Variable` and `Constant` respectively to transmute them in-place.

## v0.1.1 (2020-02-11)
- Removes some type annotations to allow compatibility with Python 3.5.

## v0.1.0 (2020-02-11)
- Adds `Node`, `Tensor` and `Graph` classes.
- Adds `BaseImporter` and `OnnxImporter` classes.
- Adds support for importing initializers in the `OnnxImporter`
- Adds `Variable` and `Constant`
- Consolidates inputs/outputs of Nodes/Tensors. Now, inputs/outputs should generally only be added to `Node`s.
- Adds `OnnxExporter` to export `Graph` to `onnx.GraphProto`
- Adds `OnnxExporter` and `OnnxImporter` to public imports
- Adds `toposort` function to `Graph`, which will topologically sort it.
- Adds `cleanup` function to `Graph`, which will remove unused nodes and tensors.
- Adds high-level API for importing/exporting `Graph`s from/to ONNX models.
- `Graph`s are now generated with a default name of `onnx_graphsurgeon`
