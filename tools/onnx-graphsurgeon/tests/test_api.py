from onnx_graphsurgeon.importers.onnx_importer import OnnxImporter
import onnx_graphsurgeon

from onnx_models import identity_model

import tempfile
import pytest
import onnx
import os

class TestApi(object):
    def setup_method(self):
        self.imported_graph = OnnxImporter.import_graph(identity_model().load().graph)


    def test_import(self):
        graph = onnx_graphsurgeon.import_onnx(onnx.load(identity_model().path))
        assert graph == self.imported_graph


    def test_export(self):
        with tempfile.NamedTemporaryFile() as f:
            onnx_model = onnx_graphsurgeon.export_onnx(self.imported_graph)
            assert onnx_model
            assert OnnxImporter.import_graph(onnx_model.graph) == self.imported_graph
