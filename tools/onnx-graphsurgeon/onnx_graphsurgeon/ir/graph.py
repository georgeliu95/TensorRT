from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Tensor, Constant, Variable
from onnx_graphsurgeon.ir.node import Node
from onnx_graphsurgeon.util import misc

from collections import OrderedDict, defaultdict
from typing import Sequence, Set, Dict, Tuple
import copy


# Functor that returns whether a Tensor has never been seen before
class UnseenTensor(object):
    def __init__(self, initial_tensors=None):
        tensors = misc.default_value(initial_tensors, [])
        self.seen_tensors = set([tensor.name for tensor in tensors])

    def __call__(self, tensor):
        # Empty tensors are never "seen"
        if tensor.is_empty():
            return True
        elif tensor.name not in self.seen_tensors:
            self.seen_tensors.add(tensor.name)
            return True
        return False


class NodeIDAdder(object):
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        # To get unique ids for each node, add an `id` attribute. This will be removed before the function returns.
        # Using the index in the node list allows the same object to count as different nodes.
        for index, node in enumerate(self.graph.nodes):
            node.id = index

    def __exit__(self, exc_type, exc_value, traceback):
        for node in self.graph.nodes:
            del node.id


class Graph(object):
    def __init__(self, nodes: Sequence[Node]=None, inputs: Sequence[Tensor]=None, outputs: Sequence[Tensor]=None, name=None, doc_string=None, opset=None):
        """
        Represents a graph containing nodes and tensors.

        Optional Args:
            nodes (Sequence[Node]): A list of the nodes in this graph.
            inputs (Sequence[Tensor]): A list of graph input Tensors.
            outputs (Sequence[Tensor]): A list of graph output Tensors.
            name (str): The name of the graph. Defaults to "onnx_graphsurgeon".
            doc_string (str): A doc_string for the graph. Defaults to "".
        """
        self.nodes = misc.default_value(nodes, [])
        self.inputs = misc.default_value(inputs, [])
        self.outputs = misc.default_value(outputs, [])

        self.name = misc.default_value(name, "onnx_graphsurgeon")
        self.doc_string = misc.default_value(doc_string, "")
        self.opset = misc.default_value(opset, 11)
        # Printing graphs can be very expensive
        G_LOGGER.ultra_verbose(lambda: "Created Graph: {:}".format(self))


    def __eq__(self, other: "Graph"):
        nodes_match = len(self.nodes) == len(other.nodes) and all([node == other_node for node, other_node in zip(self.nodes, other.nodes)])
        inputs_match = len(self.inputs) == len(other.inputs) and all([inp == other_inp for inp, other_inp in zip(self.inputs, other.inputs)])
        outputs_match = len(self.outputs) == len(other.outputs) and all([out == other_out for out, other_out in zip(self.outputs, other.outputs)])
        return nodes_match and inputs_match and outputs_match


    def node_ids(self):
        """
        Returns a context manager that supplies unique integer IDs for Nodes in the Graph.

        Example:
            with graph.node_ids():
                assert graph.nodes[0].id != graph.nodes[1].id

        Returns:
            NodeIDAdder: A context manager that supplies unique integer IDs for Nodes.
        """
        return NodeIDAdder(self)


    def _get_node_id(self, node):
        try:
            return node.id
        except AttributeError:
            G_LOGGER.critical("Encountered a node not in the graph:\n{:}.\n\nTo fix this, please append the node to this graph's `nodes` attribute.".format(node))


    # Returns a list of node ids of used nodes, and a list of used tensors.
    def _get_used_node_ids(self):
        used_node_ids = set()
        # Traverse backwards from outputs to find all used nodes.
        ignore_seen = UnseenTensor()
        used_tensors = list(filter(ignore_seen, self.outputs))

        index = 0
        while index < len(used_tensors):
            used_tensor = used_tensors[index]
            index += 1
            for node in used_tensor.inputs:
                used_node_ids.add(self._get_node_id(node))
                used_tensors.extend(filter(ignore_seen, node.inputs))
        return used_node_ids, used_tensors


    def cleanup(self, remove_unused_node_outputs=True):
        """
        Removes unused nodes and tensors from the graph.
        A node or tensor is considered unused if it does not contribute to any of the graph outputs.

        Note: This function will never modify graph output tensors.

        Optional Args:
            remove_unused_node_outputs (bool): Whether to remove unused output tensors of nodes. This will never remove
                empty tensor outputs. If this is set to False, outputs of nodes kept in the graph will not be modified.

        Returns:
            self
        """
        with self.node_ids():
            used_node_ids, used_tensors = self._get_used_node_ids()

            inputs = []
            for inp in self.inputs:
                if inp in used_tensors:
                    inputs.append(inp)
                else:
                    G_LOGGER.debug("Removing unused input: {:}".format(inp))
            self.inputs = inputs

            nodes = []
            for node in self.nodes:
                if self._get_node_id(node) in used_node_ids:
                    nodes.append(node)
                else:
                    node.inputs.clear()
                    node.outputs.clear()
                    G_LOGGER.verbose("Removing unused node: {:}".format(node))

            # Last pass to remove any hanging tensors - tensors without outputs
            if remove_unused_node_outputs:
                graph_output_names = set([tensor.name for tensor in self.outputs])
                for node in nodes:
                    def is_hanging_tensor(tensor):
                        return not tensor.is_empty() and len(tensor.outputs) == 0 and tensor.name not in graph_output_names

                    [node.outputs.remove(out) for out in node.outputs if is_hanging_tensor(out)]

            self.nodes = nodes
            return self


    def toposort(self):
        """
        Topologically sort the graph in place.

        Returns:
            self
        """
        # Keeps track of a node and it's level in the graph hierarchy. 0 corresponds to an input node, N corresponds to a node with N layers of inputs.
        class HierarchyDescriptor(object):
            def __init__(self, node=None, level=None):
                self.node = node
                self.level = level


            def __lt__(self, other):
                return self.level < other.level

        hierarchy_levels = {} # Dict[int, HierarchyDescriptor]

        def get_hierarchy_level(node):
            # Return all nodes that contribute to this node.
            def get_input_nodes(node):
                inputs = {}
                for tensor in node.inputs:
                    for node in tensor.inputs:
                        inputs[self._get_node_id(node)] = node
                return inputs.values()

            if self._get_node_id(node) in hierarchy_levels:
                return hierarchy_levels[self._get_node_id(node)].level

            # The level of a node is the level of it's highest input + 1.
            try:
                max_input_level = max([get_hierarchy_level(input_node) for input_node in get_input_nodes(node)] + [-1])
            except RecursionError:
                G_LOGGER.critical("Cycle detected in graph! Are there tensors with duplicate names in the graph?")

            return max_input_level + 1

        with self.node_ids():
            for node in self.nodes:
                hierarchy_levels[self._get_node_id(node)] = HierarchyDescriptor(node, level=get_hierarchy_level(node))

        self.nodes = [hd.node for hd in sorted(hierarchy_levels.values())]
        return self


    def tensors(self, check_duplicates=False):
        """
        Creates a tensor map of all the tensors in this graph by walking over all nodes. Empty tensors are omitted from this map. The graph must not contain tensors with duplicate names.

        Tensors are guaranteed to be in order of the nodes in the graph. Hence, if the graph is topologically sorted, the tensor map will be too.

        Optional Args:
            check_duplicates (bool): Whether to fail if multiple tensors with the same name are encountered.

        Raises:
            OnnxGraphSurgeonException: If check_duplicates is True, and multiple distinct tensors in the graph share the same name.

        Returns:
            OrderedDict[str, Tensor]: A mapping of tensor names to tensors.
        """
        tensor_map = OrderedDict()

        def add_to_tensor_map(tensor):
            if not tensor.is_empty():
                if check_duplicates and tensor.name in tensor_map and not (tensor_map[tensor.name] is tensor):
                    G_LOGGER.critical("Found distinct tensors that share the same name:\n[id: {:}] {:}\n[id: {:}] {:}"
                        .format(id(tensor_map[tensor.name]), tensor_map[tensor.name], id(tensor), tensor))
                tensor_map[tensor.name] = tensor

        for node in self.nodes:
            for tensor in node.inputs + node.outputs:
                add_to_tensor_map(tensor)
        return tensor_map


    def fold_constants(self):
        """
        Folds constants in-place in the graph. The graph must be topologically sorted prior to calling this function (see `toposort()`).

        NOTE: This function will not remove constants after folding them. In order to get rid of these hanging nodes, you can run the `cleanup()` function.

        NOTE: Due to how this is implemented, the graph must be exportable to ONNX, and evaluable in ONNX Runtime.

        Returns:
            self
        """
        from onnx_graphsurgeon.api.api import export_onnx
        import onnxruntime
        import onnx

        temp_graph = copy.deepcopy(self)

        # Since the graph is topologically sorted, this should find all constant nodes in the graph.
        graph_constants = {tensor.name: tensor for tensor in temp_graph.tensors().values() if isinstance(tensor, Constant)}
        for node in temp_graph.nodes:
            if all([inp.name in graph_constants for inp in node.inputs]):
                graph_constants.update({out.name: out for out in node.outputs})

        # Next build a graph with just the constants, and evaluate - no need to evaluate constants
        outputs_to_evaluate = [tensor for tensor in graph_constants.values() if isinstance(tensor, Variable)]
        output_names = [out.name for out in outputs_to_evaluate]

        temp_graph.outputs = outputs_to_evaluate
        temp_graph.cleanup()

        sess = onnxruntime.InferenceSession(export_onnx(temp_graph).SerializeToString())
        constant_values = sess.run(output_names, {})

        # Finally, replace the Variables in the original graph with constants.
        graph_tensors = self.tensors()
        for name, values in zip(output_names, constant_values):
            graph_tensors[name].to_constant(values)
            graph_tensors[name].inputs.clear() # Constants do not need inputs

        return self


    def __deepcopy__(self, memo):
        """
        Makes a deep copy of this graph.
        """
        # First, reconstruct each tensor in the graph, but with no inputs or outputs
        tensor_map = self.tensors()
        new_tensors = {name: tensor.copy() for name, tensor in tensor_map.items()}

        # Next, copy nodes, and update inputs/outputs
        new_nodes = []
        for node in self.nodes:
            new_node = node.copy(inputs=[new_tensors[inp.name] for inp in node.inputs], outputs=[new_tensors[out.name] for out in node.outputs])
            new_nodes.append(new_node)

        new_graph_inputs = [new_tensors[inp.name] for inp in self.inputs]
        new_graph_outputs = [new_tensors[out.name] for out in self.outputs]
        return Graph(nodes=new_nodes, inputs=new_graph_inputs, outputs=new_graph_outputs, name=copy.deepcopy(self.name, memo), doc_string=copy.deepcopy(self.doc_string, memo))


    def __str__(self):
        nodes_str = "\n".join([str(node) for node in self.nodes])
        return "Graph {:} (Opset: {:})\nInputs: {:}\nNodes: {:}\nOutputs: {:}".format(self.name, self.opset, self.inputs, nodes_str, self.outputs)


    def __repr__(self):
        return self.__str__()
