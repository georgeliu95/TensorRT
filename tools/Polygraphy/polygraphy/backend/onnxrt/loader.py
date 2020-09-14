from polygraphy.backend.base import BaseLoadModel
from polygraphy.util import misc

import onnxruntime
misc.log_module_info(onnxruntime)

class SessionFromOnnxBytes(BaseLoadModel):
    def __init__(self, model_bytes):
        """
        Functor that builds an ONNX-Runtime inference session.

        Args:
            model_bytes (Callable() -> bytes): A loader that can supply a serialized ONNX model.
        """
        self._model_bytes = model_bytes


    def __call__(self):
        """
        Builds an ONNX-Runtime inference session.

        Returns:
            onnxruntime.InferenceSession: The inference session.
        """
        model_bytes, _ = misc.try_call(self._model_bytes)
        return onnxruntime.InferenceSession(model_bytes)
