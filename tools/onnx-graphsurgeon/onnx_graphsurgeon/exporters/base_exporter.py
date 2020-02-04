from onnx_graphsurgeon.ir.graph import Graph

class BaseExporter(object):
    @staticmethod
    def export_graph(graph: Graph):
        """
        Export a graph from some destination graph.

        Args:
            graph (Graph): The source graph to export.

        Returns:
            object: The exported graph. For example, this might be an onnx.GraphProto
        """
        raise NotImplementedError("BaseExporter is an abstract class")
