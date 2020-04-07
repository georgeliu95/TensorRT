# Contains high-level API functions.
from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.graph import Graph


def import_onnx(onnx_model: "onnx.ModelProto") -> Graph:
    """
    Import an onnx-graphsurgeon Graph from the provided ONNX model.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.

    Returns:
        Graph: A corresponding onnx-graphsurgeon Graph.
    """
    from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter

    return OnnxImporter.import_graph(onnx_model.graph, opset=OnnxImporter.get_opset(onnx_model))


def export_onnx(graph: Graph, **kwargs) -> "onnx.ModelProto":
    """
    Exports an onnx-graphsurgeon Graph to an ONNX model.

    Args:
        graph (Graph): The graph to export

    Optional Args:
        **kwargs: Additional arguments to onnx.helper.make_model

    Returns:
        onnx.ModelProto: A corresponding ONNX model.
    """
    from onnx_graphsurgeon.exporters.onnx_exporter import OnnxExporter
    import onnx

    onnx_graph = OnnxExporter.export_graph(graph)

    if "opset_imports" not in kwargs:
        kwargs["opset_imports"] = [onnx.helper.make_opsetid("", graph.opset)]

    return onnx.helper.make_model(onnx_graph, **kwargs)