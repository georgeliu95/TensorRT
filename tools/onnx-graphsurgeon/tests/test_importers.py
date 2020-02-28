from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter
from onnx_graphsurgeon.logger.logger import G_LOGGER

from onnx_models import identity_model, lstm_model, scan_model, dim_param_model

from collections import OrderedDict
import onnx.numpy_helper
import numpy as np
import pytest
import onnx

G_LOGGER.severity = G_LOGGER.ULTRA_VERBOSE

class TestOnnxImporter(object):
    def test_import_tensor(self):
        name = "test0"
        shape = [1, 2, 3, 4]
        onnx_tensor = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        tensor = OnnxImporter.import_tensor(onnx_tensor)
        assert tensor.name == name
        assert tensor.dtype == np.float32
        assert tensor.shape == shape

    # TODO: Test all attribute types - missing graph
    def test_import_node(self):
        op = "Test"
        inputs = ["x"]
        outputs = ["y"]
        float_attr = 4.0
        int_attr = 10
        str_attr = "constant"
        tensor_vals = np.ones(shape=(1, 2, 3, 4), dtype=np.float32)
        tensor_attr = onnx.numpy_helper.from_array(tensor_vals)
        floats_attr = [1.0, 2.0, 3.0, 4.0]
        ints_attr = [4, 3, 2, 1]
        strings_attr = ["constant", "and", "variable"]

        onnx_node = onnx.helper.make_node(op, inputs, outputs, float_attr=float_attr, int_attr=int_attr, str_attr=str_attr, tensor_attr=tensor_attr, floats_attr=floats_attr, ints_attr=ints_attr, strings_attr=strings_attr)
        node = OnnxImporter.import_node(onnx_node, OrderedDict())
        assert node.op == op
        assert node.attrs["float_attr"] == float_attr
        assert node.attrs["int_attr"] == int_attr
        assert node.attrs["str_attr"] == str_attr
        # Tensor should turn into a ConstantTensor
        assert np.all(node.attrs["tensor_attr"].values == tensor_vals)
        assert node.attrs["floats_attr"] == floats_attr
        assert node.attrs["ints_attr"] == ints_attr
        assert node.attrs["strings_attr"] == strings_attr


    @pytest.mark.parametrize("model", [identity_model(), lstm_model(), scan_model()])
    def test_import_graph(self, model):
        graph = OnnxImporter.import_graph(model.load().graph)
        model.assert_equal(graph)


    def test_import_graph_tensor_map_preserved(self):
        model = identity_model()
        tensor_map = OrderedDict()
        graph = OnnxImporter.import_graph(model.load().graph, tensor_map=tensor_map)
        assert len(tensor_map) == 0
        model.assert_equal(graph)

    def test_import_graph_with_initializer(self):
        model = lstm_model()
        graph = OnnxImporter.import_graph(model.load().graph)
        model.assert_equal(graph)


    def test_import_graph_with_dim_param(self):
        model = dim_param_model()
        graph = OnnxImporter.import_graph(model.load().graph)
        model.assert_equal(graph)
