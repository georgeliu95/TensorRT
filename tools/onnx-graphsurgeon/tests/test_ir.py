from onnx_graphsurgeon.logger.logger import G_LOGGER
from onnx_graphsurgeon.ir.tensor import Tensor, ConstantTensor, VariableTensor
from onnx_graphsurgeon.ir.graph import Graph
from onnx_graphsurgeon.ir.node import Node

import numpy as np
import pytest
import onnx
import copy

G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE

class TestTensor(object):
    def setup_method(self):
        self.tensor = Tensor(name="test_tensor")
        self.input_node = Node(op="Add", outputs=[self.tensor])
        self.output_node = Node(op="Add", inputs=[self.tensor])

    def test_set_inputs_updates_old_inputs(self):
        dummy = Node(op="dummy")
        self.tensor.inputs = [dummy]
        assert len(self.input_node.outputs) == 0
        assert dummy.outputs[0] == self.tensor

    def test_set_outputs_updates_old_outputs(self):
        dummy = Node(op="dummy")
        self.tensor.outputs = [dummy]
        assert len(self.output_node.inputs) == 0
        assert dummy.inputs[0] == self.tensor

    def test_can_copy_inputs_from_other_node(self):
        tensor = Tensor(name="other_test_tensor")
        tensor.inputs = self.tensor.inputs
        assert tensor.inputs == self.tensor.inputs

    def test_can_copy_outputs_from_other_node(self):
        tensor = Tensor(name="other_test_tensor")
        tensor.outputs = self.tensor.outputs
        assert tensor.outputs == self.tensor.outputs


class TestVariableTensor(object):
    def setup_method(self):
        self.node = Node(op="Add")
        self.tensor = VariableTensor(name="test_tensor", dtype=np.float32, shape=(1, 3, 224, 224))
        self.node.outputs.append(self.tensor)

    def test_equals(self):
        assert self.tensor == self.tensor

    def test_equals_name_mismatch(self):
        tensor = VariableTensor(name="test_tensor0", dtype=np.float32, shape=(1, 3, 224, 224))
        assert not self.tensor == tensor

    def test_equals(self):
        assert self.tensor == self.tensor

    def test_can_convert_in_place_to_constant(self):
        self.tensor.make_constant(values=np.ones((1, 3, 5, 5), dtype=np.float64))
        assert isinstance(self.tensor, ConstantTensor)
        assert isinstance(self.node.outputs[0], ConstantTensor)
        assert self.tensor.shape == (1, 3, 5, 5)
        assert self.tensor.dtype == np.float64
        assert np.all(self.node.outputs[0].values == self.tensor.values)


class TestConstantTensor(object):
    def setup_method(self):
        self.node = Node(op="Add")
        self.tensor = ConstantTensor(name="test_tensor", values=np.ones((1, 3, 5, 5), dtype=np.float64))
        self.node.outputs.append(self.tensor)

    def test_can_get_shape(self):
        assert self.tensor.shape == (1, 3, 5, 5)

    def test_can_get_dtype(self):
        assert self.tensor.dtype == np.float64

    def test_can_convert_in_place_to_variable(self):
        self.tensor.make_variable(dtype=np.float32, shape=(1, 3, 224, 224))
        assert isinstance(self.tensor, VariableTensor)
        assert isinstance(self.node.outputs[0], VariableTensor)
        assert self.tensor.dtype == np.float32
        assert self.tensor.shape == (1, 3, 224, 224)
        assert self.node.outputs[0].dtype == self.tensor.dtype
        assert self.node.outputs[0].shape == self.tensor.shape



class TestNode(object):
    def setup_method(self):
        self.input_tensor = Tensor(name="x")
        self.output_tensor = Tensor(name="y")
        self.node = Node(op="Add", name="Test", inputs=[self.input_tensor], outputs=[self.output_tensor])

    def test_equals(self):
        assert self.node == self.node

    def test_equals_name_mismatch(self):
        node = Node(op="Add", name="OtherTest")
        assert not self.node == node

    def test_equals_op_mismatch(self):
        node = Node(op="Subtract", name="Test")
        assert not self.node == node

    def test_equals_num_inputs_mismatch(self):
        node = Node(op="Subtract", name="Test")
        assert not self.node == node

    def test_equals(self):
        assert self.node == self.node

    def test_equals_inputs_mismatch(self):
        tensor = Tensor(name="other_tensor")
        assert not self.input_tensor == tensor

        node = Node(op="Add", name="Test", inputs=[tensor])
        assert not self.node == node

    def test_set_inputs_updates_old_inputs(self):
        dummy = Tensor(name="dummy")
        self.node.inputs = [dummy]
        assert len(self.input_tensor.outputs) == 0
        assert dummy.outputs[0] == self.node

    def test_set_outputs_updates_old_outputs(self):
        dummy = Tensor(name="dummy")
        self.node.outputs = [dummy]
        assert len(self.output_tensor.inputs) == 0
        assert dummy.inputs[0] == self.node

    def test_can_copy_inputs_from_other_node(self):
        node = Node(op="Subtract")
        node.inputs = self.node.inputs
        assert node.inputs == self.node.inputs

    def test_can_copy_outputs_from_other_node(self):
        node = Node(op="Subtract")
        node.outputs = self.node.outputs
        assert node.outputs == self.node.outputs


class TestNodeIO(object):
    def setup_method(self, field_names):
        self.tensors = [VariableTensor(name="test_tensor_{:}".format(i), dtype=np.float32, shape=(1, 3, 224, 224)) for i in range(10)]
        self.node = Node(op="Dummy")

    def get_lists(self, field_names):
        return getattr(self.node, field_names[0]), field_names[1]

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_append(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[0])
        assert nlist[0] == self.tensors[0]
        assert getattr(self.tensors[0], tensor_field)[0] == self.node

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_extend(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.extend(self.tensors)
        for tensor in self.tensors:
            assert tensor in nlist
            assert getattr(tensor, tensor_field)[0] == self.node

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_insert(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[1])
        nlist.insert(0, self.tensors[0])
        assert nlist[0] == self.tensors[0]
        assert getattr(self.tensors[0], tensor_field)[0] == self.node

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_remove(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[0])
        nlist.remove(self.tensors[0])
        assert len(nlist) == 0
        assert len(getattr(self.tensors[0], tensor_field)) == 0

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_pop(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[0])
        tensor = nlist.pop()
        assert len(nlist) == 0
        assert len(getattr(tensor, tensor_field)) == 0

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_pop_index(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.extend(self.tensors)
        tensor = nlist.pop(1)
        assert self.tensors[1] not in nlist
        assert len(getattr(tensor, tensor_field)) == 0

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_del_index(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.extend(self.tensors)
        tensor = nlist[1]
        del nlist[1]
        assert self.tensors[1] not in nlist
        assert len(getattr(tensor, tensor_field)) == 0

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_clear(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.extend(self.tensors)
        nlist.clear()
        assert len(nlist) == 0
        assert all([len(getattr(tensor, tensor_field)) == 0 for tensor in self.tensors])

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_add(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist = nlist + self.tensors
        for tensor in self.tensors:
            assert tensor in nlist
            assert getattr(tensor, tensor_field)[0] == self.node

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_iadd(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist += self.tensors
        for tensor in self.tensors:
            assert tensor in nlist
            assert getattr(tensor, tensor_field)[0] == self.node

    @pytest.mark.parametrize("field_names", [("inputs", "outputs"), ("outputs", "inputs")])
    def test_setitem(self, field_names):
        nlist, tensor_field = self.get_lists(field_names)
        nlist.append(self.tensors[0])
        new_tensor = Tensor("new_tensor")
        nlist[0] = new_tensor
        assert nlist[0] == new_tensor
        assert len(getattr(self.tensors[0], tensor_field)) == 0
        assert getattr(new_tensor, tensor_field)[0] == self.node


def build_basic_graph():
    inputs = [Tensor(name="x")]
    outputs = [Tensor(name="y")]
    nodes = [
        Node(op="Add", name="Test", inputs=inputs, outputs=outputs),
    ]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs)


def build_two_layer_graph():
    inputs = [Tensor(name="x")]
    intermediate_tensor = Tensor(name="intermediate")
    outputs = [Tensor(name="y")]
    nodes = [
        Node(op="Add", name="Test0", inputs=inputs, outputs=[intermediate_tensor]),
        Node(op="Add", name="Test1", inputs=[intermediate_tensor], outputs=outputs),
    ]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs)


def build_two_layer_graph_multiple_io():
    inputs = [Tensor(name="x0"), Tensor(name="x1")]
    intermediate_tensor = Tensor(name="intermediate")
    outputs = [Tensor(name="y0"), Tensor(name="y1")]
    nodes = [
        Node(op="Add", name="Test0", inputs=inputs, outputs=[intermediate_tensor]),
        Node(op="Add", name="Test1", inputs=[intermediate_tensor], outputs=outputs),
    ]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs)


GRAPH_TEST_CASES = [
    build_basic_graph(),
    build_two_layer_graph(),
    build_two_layer_graph_multiple_io(),
]


def toposort_linear_graph():
    inputs = [Tensor(name="x")]
    intermediate0 = Tensor(name="intermediate0")
    intermediate1 = Tensor(name="intermediate1")
    intermediate2 = Tensor(name="intermediate2")
    outputs = [Tensor(name="y")]
    # Nodes are NOT in topo order.
    nodes = [
        Node(op="Add", name="Test0", inputs=inputs, outputs=[intermediate0]),
        Node(op="Add", name="Test2", inputs=[intermediate1], outputs=[intermediate2]),
        Node(op="Add", name="Test3", inputs=[intermediate2], outputs=outputs),
        Node(op="Add", name="Test1", inputs=[intermediate0], outputs=[intermediate1]),
    ]
    expected_node_order = [nodes[0], nodes[3], nodes[1], nodes[2]]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs), expected_node_order


# Graph structure:
# x
# |
# Test0 -> out0 (graph output)
# |
# out0
# |
# Test1 -> out1 (graph output)
# |
# out1
# |
# Test2 -> out2 (graph_output)
def toposort_multi_tier_output_graph():
    inputs = [Tensor(name="x")]
    outputs = [Tensor(name="out0"), Tensor(name="out1"), Tensor(name="out2")]
    out0, out1, out2 = outputs
    nodes = [
        Node(op="Add", name="Test2", inputs=[out1], outputs=[out2]),
        Node(op="Add", name="Test0", inputs=inputs, outputs=[out0]),
        Node(op="Add", name="Test1", inputs=[out0], outputs=[out1]),
    ]
    expected_node_order = [nodes[1], nodes[2], nodes[0]]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs), expected_node_order


# Graph structure:
# x2  x1
# |   |
# Test0
# |
# int0  x0
# |    /
# Test1
# |
# int1  x3
# |    /
# Test2 -> out (graph_output)
def toposort_multi_tier_input_graph():
    inputs = [Tensor(name="x0"), Tensor(name="x1"), Tensor(name="x2"), Tensor(name="x3")]
    int0, int1 = [Tensor(name="intermediate0"), Tensor(name="intermediate1")]
    outputs = [Tensor(name="out")]
    x0, x1, x2, x3 = inputs
    nodes = [
        Node(op="Add", name="Test2", inputs=[int1, x3], outputs=outputs),
        Node(op="Add", name="Test0", inputs=[x2, x1], outputs=[int0]),
        Node(op="Add", name="Test1", inputs=[int0, x0], outputs=[int1]),
    ]
    expected_node_order = [nodes[1], nodes[2], nodes[0]]
    return Graph(nodes=nodes, inputs=inputs, outputs=outputs), expected_node_order


TOPOSORT_TEST_CASES = [
    toposort_linear_graph(),
    toposort_multi_tier_output_graph(),
    toposort_multi_tier_input_graph(),
]

class TestGraph(object):
    @pytest.mark.parametrize("graph", GRAPH_TEST_CASES)
    def test_get_used_node_ids(self, graph):
        graph_used_nodes = copy.copy(graph.nodes)
        graph_used_tensors = copy.copy(list(graph.generate_tensor_map().values()))

        unused_tensor = Tensor(name="Unused")
        unused_node = Node(op="Unused", inputs=[graph.inputs[0]], outputs=[unused_tensor])
        graph.nodes.append(unused_node)

        with graph.node_ids():
            used_node_ids, used_tensors = graph._get_used_node_ids()
            assert len(used_node_ids) == len(graph.nodes) - 1
            assert all([node.id in used_node_ids for node in graph_used_nodes])
            assert unused_node.id not in used_node_ids
            assert unused_tensor not in used_tensors
            assert all([used_tensor in used_tensors for used_tensor in graph_used_tensors])


    @pytest.mark.parametrize("toposort_test_case", TOPOSORT_TEST_CASES)
    def test_topologically_sort(self, toposort_test_case):
        graph, expected_node_order = toposort_test_case
        assert graph.nodes != expected_node_order
        graph.toposort()
        assert graph.nodes == expected_node_order


    def test_cleanup_multi_tier(self):
        graph, _ = toposort_multi_tier_output_graph()
        tensor = graph.outputs.pop()
        unused_node = tensor.inputs[0]
        graph.cleanup() # Should remove just the Test2 node as out1 is still an output.
        assert unused_node not in graph.nodes
        assert len(graph.nodes) == 2
        assert len(graph.outputs) == 2

        tensor_map = graph.generate_tensor_map()
        assert tensor.name not in tensor_map


    def test_cleanup_intermediate_tensors(self):
        graph, _  = toposort_linear_graph()
        graph.toposort()
        graph_output = graph.outputs[0]

        dummy = Tensor("dummy")
        # Add unused tensor to a node in the middle of the graph.
        # Since it does not contribute to graph outputs, it should be removed.
        graph.nodes[1].outputs.append(dummy)

        graph.cleanup()
        assert dummy not in graph.nodes[1].outputs
        assert graph.outputs[0] == graph_output # Graoh outputs will never be removed


    def test_cleanup_independent_path(self):
        graph, _ = toposort_linear_graph()
        # Build out a path totally unrelated to rest of the graph
        indep0 = Tensor(name="indep0")
        indep1 = Tensor(name="indep1")
        node = Node(op="IndepTest", inputs=[indep0], outputs=[indep1])
        graph.inputs.append(indep0) # Unused inputs should be removed as well
        graph.nodes.append(node)
        graph.cleanup()
        assert indep0 not in graph.inputs
        assert node not in graph.nodes

        tensor_map = graph.generate_tensor_map()
        assert indep0.name not in tensor_map
        assert indep1.name not in tensor_map


    def test_deep_copy(self):
        def make_graph():
            graph, _ = toposort_multi_tier_output_graph()
            graph.outputs.pop()
            return graph

        graph = make_graph()
        new_graph = copy.deepcopy(graph)
        assert graph == new_graph

        # Running cleanup on the first graph should not affect the copy
        graph.cleanup()
        assert graph != new_graph
        assert new_graph == make_graph()


    def test_fold_constants(self):
        # Graph:
        # c = (a + b)
        # output = input + c
        # Should fold to:
        # output = input + c
        inp = VariableTensor("input", shape=(1, 3), dtype=np.float32)
        a = ConstantTensor("a", values=np.ones(shape=(1, 3), dtype=np.float32))
        b = ConstantTensor("b", values=np.ones(shape=(1, 3), dtype=np.float32))
        c = VariableTensor("c", shape=(1, 3), dtype=np.float32)
        out = VariableTensor("output", shape=(1, 3), dtype=np.float32)

        nodes = [
            Node("Add", inputs=[a, b], outputs=[c]),
            Node("Add", inputs=[inp, c], outputs=[out]),
        ]
        graph = Graph(nodes=nodes, inputs=[inp], outputs=[out])

        graph.fold_constants().cleanup()

        # Extra node should be removed
        assert len(graph.nodes) == 1
        assert graph.nodes[0].inputs[0] == inp
        assert graph.nodes[0].inputs[1] == c
        # Value should be computed correctly
        assert np.all(graph.nodes[0].inputs[1].values == np.ones(shape=(1, 3), dtype=np.float32) * 2)


    def test_fold_constants_one_hop(self):
        # Graph:
        # c = (a + b)
        # e = (c + d)
        # output = input + e
        # Should fold to:
        # output = input + e
        inp = VariableTensor("input", shape=(1, 3), dtype=np.float32)
        a = ConstantTensor("a", values=np.ones(shape=(1, 3), dtype=np.float32))
        b = ConstantTensor("b", values=np.ones(shape=(1, 3), dtype=np.float32))
        c = VariableTensor("c", shape=(1, 3), dtype=np.float32)
        d = ConstantTensor("d", values=np.ones(shape=(1, 3), dtype=np.float32))
        e = VariableTensor("e", shape=(1, 3), dtype=np.float32)
        out = VariableTensor("output", shape=(1, 3), dtype=np.float32)

        nodes = [
            Node("Add", inputs=[a, b], outputs=[c]),
            Node("Add", inputs=[c, d], outputs=[e]),
            Node("Add", inputs=[inp, e], outputs=[out]),
        ]

        graph = Graph(nodes=nodes, inputs=[inp], outputs=[out])

        graph.fold_constants().cleanup()

        # Extra nodes should be removed
        assert len(graph.nodes) == 1
        assert graph.nodes[0].inputs[0] == inp
        assert graph.nodes[0].inputs[1] == e
        # Value should be computed correctly
        assert np.all(graph.nodes[0].inputs[1].values == np.ones(shape=(1, 3), dtype=np.float32) * 3)
