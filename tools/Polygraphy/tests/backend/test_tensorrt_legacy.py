from polygraphy.backend.trt_legacy import TrtLegacyRunner, LoadNetworkFromUff, ConvertToUff, ParseNetworkFromOnnxLegacy

from tests.models.meta import TF_MODELS, ONNX_MODELS

import numpy as np
import pytest


@pytest.mark.parametrize("fp16", [True, False], ids=lambda x: "fp16" if x else "")
@pytest.mark.parametrize("tf32", [True, False], ids=lambda x: "tf32" if x else "")
def test_uff_identity(fp16, tf32):
    model = TF_MODELS["identity"]
    loader = model.loader
    with TrtLegacyRunner(LoadNetworkFromUff(ConvertToUff(loader)), fp16=fp16, tf32=tf32) as runner:
        assert runner.is_active
        feed_dict = {"Input": np.random.random_sample(size=(1, 15, 25, 30)).astype(np.float32)}
        outputs = runner.infer(feed_dict)
        assert np.all(outputs["Identity_2"] == feed_dict["Input"])
    assert not runner.is_active


def test_can_construct_onnx_loader():
    model = ONNX_MODELS["identity"].path
    loader = ParseNetworkFromOnnxLegacy(model)
