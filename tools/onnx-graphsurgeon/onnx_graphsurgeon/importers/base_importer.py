from onnx_graphsurgeon.ir.graph import Graph

class BaseImporter(object):
    @staticmethod
    def import_graph(graph) -> Graph:
        """
        Import a graph from some source graph.

        Args:
            graph (object): The source graph to import. For example, this might be an onnx.GraphProto.

        Returns:
            Graph: The equivalent onnx-graphsurgeon graph.
        """
        raise NotImplementedError("BaseImporter is an abstract class")
