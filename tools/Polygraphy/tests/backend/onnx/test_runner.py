from polygraphy.backend.onnx import OnnxTfRunner, OnnxFromPath, BytesFromOnnx

from tests.models.meta import ONNX_MODELS


class TestOnnxTfRunner(object):
    def test_can_name_runner(self):
        NAME = "runner"
        runner = OnnxTfRunner(None, name=NAME)
        assert runner.name == NAME


    def test_basic(self):
        model = ONNX_MODELS["identity"]
        with OnnxTfRunner(OnnxFromPath(model.path)) as runner:
            assert runner.is_active
            model.check_runner(runner)
        assert not runner.is_active
