from polygraphy.common import constants, TensorMetadata
from polygraphy.logger.logger import G_LOGGER
from polygraphy.util import misc

from collections import OrderedDict
import copy


def check_model(model):
    try:
        import onnx
    except:
        G_LOGGER.warning("Could not import onnx module, skipping model check")

    try:
        onnx.checker.check_model(model)
        G_LOGGER.verbose("ONNX Checker Passed")
    except onnx.checker.ValidationError as err:
        G_LOGGER.warning("ONNX Checker exited with an error:\n{:}".format(err))
    finally:
        return model


def infer_shapes(model):
    try:
        import onnx.shape_inference
    except:
        G_LOGGER.warning("Could not import onnx.shape_inference module, skipping shape inference")

    try:
        model = onnx.shape_inference.infer_shapes(model)
        G_LOGGER.verbose("ONNX Shape Inference completed successfully")
    except Exception as err:
        G_LOGGER.warning("ONNX shape inference exited with an error:\n{:}".format(err))
    finally:
        return model


def all_tensor_names(model):
    all_outputs = [output for node in model.graph.node if node.op_type != "Constant" for output in node.output]
    all_outputs = misc.unique_list(all_outputs)
    return all_outputs


def mark_outputs(model, outputs):
    import onnx

    # Clear the old outputs
    while model.graph.output:
        model.graph.output.pop()

    all_outputs = all_tensor_names(model)
    all_outputs_set = set(all_outputs)

    out_tensors = []
    not_found = set()
    for output in outputs:
        if output in all_outputs_set:
            out_tensors.append(onnx.helper.make_empty_tensor_value_info(output))
        else:
            not_found.add(output)

    if not_found:
        G_LOGGER.critical("The following outputs: {:} were not found in the network. "
                          "Note: Available tensors: {:}".format(not_found, all_outputs))

    model.graph.output.extend(out_tensors)
    return model


def mark_layerwise(model):
    # Add all non-constant node outputs as graph outputs
    model = mark_outputs(model, all_tensor_names(model))
    return model


def get_shape(tensor):
    import onnx

    shape = []
    if isinstance(tensor, onnx.TensorProto):
        shape = tensor.dims
    else:
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(-1)
            else:
                shape.append(dim.dim_value)
    return shape


def get_dtype(tensor):
    import onnx

    if isinstance(tensor, onnx.TensorProto):
        onnx_type = tensor.data_type
    else:
        onnx_type = tensor.type.tensor_type.elem_type
    if onnx_type in onnx.mapping.TENSOR_TYPE_TO_NP_TYPE:
        return onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx_type]
    return None


def get_tensor_metadata(tensors):
    metadata = TensorMetadata()
    for tensor in tensors:
        metadata.add(name=tensor.name, dtype=get_dtype(tensor), shape=get_shape(tensor))
    return metadata


def get_input_metadata(graph):
    # Some "inputs" are actually weights with initalizers, so we need to eliminate those.
    initializer_names = set([tensor.name for tensor in graph.initializer])
    input_tensors = [tensor for tensor in graph.input if tensor.name not in initializer_names]
    return get_tensor_metadata(input_tensors)


def get_output_metadata(graph):
    return get_tensor_metadata(graph.output)


def str_from_onnx_graph(graph, layer_info, attr_info, tensors, indent_level=0):
    import onnx

    input_metadata = get_input_metadata(graph)
    output_metadata = get_output_metadata(graph)
    initializer_metadata = get_tensor_metadata(graph.initializer)

    graph_type = "Graph" if indent_level == 0 else "Subgraph"

    onnx_str = ""
    onnx_str += "---- {:} {:} Inputs ----\n{:}\n\n".format(len(input_metadata), graph_type, input_metadata)
    onnx_str += "---- {:} {:} Outputs ----\n{:}\n\n".format(len(output_metadata), graph_type, output_metadata)

    onnx_str += "---- {:} Initializers ----\n".format(len(initializer_metadata))
    if layer_info:
        onnx_str += str(initializer_metadata)
    else:
        onnx_str += "(Use --layer-info to display)"
    onnx_str += "\n\n"

    # Subgraph inputs should remain separate from each other, hence copy the tensors map
    tensors = copy.copy(tensors)
    tensors.update(get_tensor_metadata(graph.value_info))
    tensors.update(initializer_metadata)
    tensors.update(input_metadata)
    tensors.update(output_metadata)

    def metadata_from_names(names):
        metadata = TensorMetadata()
        for name in names:
            dtype = None
            shape = None
            if name in tensors:
                dtype, shape = tensors[name]
            if name in initializer_metadata:
                name = "{:} (initializer)".format(name)
            metadata.add(name=name, dtype=dtype, shape=shape)
        return metadata

    # Maps values from the AttributeType enum to their string representations, e.g., {1: "FLOAT"}
    ATTR_TYPE_MAPPING = dict(zip(onnx.AttributeProto.AttributeType.values(), onnx.AttributeProto.AttributeType.keys()))

    # Maps an ONNX attribute to the corresponding Python property
    ONNX_PYTHON_ATTR_MAPPING = {
        "FLOAT": "f",
        "INT": "i",
        "STRING": "s",
        "TENSOR": "t",
        "GRAPH": "g",
        "FLOATS": "floats",
        "INTS": "ints",
        "STRINGS": "strings",
    }

    def attrs_to_dict(attrs):
        attr_dict = OrderedDict()
        for attr in attrs:
            def process_attr(attr_str: str):
                processed = getattr(attr, ONNX_PYTHON_ATTR_MAPPING[attr_str])
                if attr_str == "STRING":
                    processed = processed.decode()
                elif attr_str == "TENSOR":
                    processed = "Tensor: {:} [dtype={:}, shape={:}]".format(processed.name, get_dtype(processed), get_shape(processed))
                elif attr_str == "GRAPH":
                    processed = "\n" + str_from_onnx_graph(processed, layer_info, attr_info, tensors, indent_level=indent_level + 2)
                elif attr_str == "FLOATS" or attr_str == "INTS":
                    # Proto hacky list to normal Python list
                    processed = [p for p in processed]
                elif attr_str == "STRINGS":
                    processed = [p.decode() for p in processed]
                return processed

            if attr.type in ATTR_TYPE_MAPPING:
                attr_str = ATTR_TYPE_MAPPING[attr.type]
                if attr_str in ONNX_PYTHON_ATTR_MAPPING:
                    attr_dict[attr.name] = process_attr(attr_str)
                else:
                    G_LOGGER.warning("Attribute of type {:} is currently unsupported. Skipping attribute.".format(attr_str))
            else:
                G_LOGGER.warning("Attribute type: {:} was not recognized. Was the graph generated with a newer IR "
                                "version than the installed `onnx` package? Skipping attribute.".format(attr.type))
        return attr_dict


    onnx_str += "---- {:} Nodes ----\n".format(len(graph.node))
    if layer_info:
        for index, node in enumerate(graph.node):
            input_info = metadata_from_names(node.input)
            output_info = metadata_from_names(node.output)

            onnx_str += "Node {:<4} | {:} [Op: {:}]\n{tab}{:}\n{tab}-> {:}\n".format(
                                index, node.name, node.op_type, input_info, output_info, tab=constants.TAB)
            if attr_info:
                attrs = attrs_to_dict(node.attribute)
                for key, val in attrs.items():
                    onnx_str += "{:}{:}.{:} = {:}\n".format(constants.TAB, node.name, key, val)
            onnx_str += "\n"
    else:
        onnx_str += "(Use --layer-info to display)"


    indent = constants.TAB * indent_level
    sep = "\n" + indent
    onnx_str = indent + sep.join(onnx_str.splitlines())
    return onnx_str


def str_from_onnx(model, layer_info=True, attr_info=True):
    def get_opset():
        try:
            return model.opset_import[0].version
        except:
            G_LOGGER.warning("Model does not contain opset information!")
            return None

    onnx_str = ""
    onnx_str += "Name: {:} | Opset: {:}\n".format(model.graph.name, get_opset())
    onnx_str += "\n"

    onnx_str += str_from_onnx_graph(model.graph, layer_info, attr_info, tensors={})
    return onnx_str
