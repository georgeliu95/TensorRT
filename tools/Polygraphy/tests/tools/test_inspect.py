import copy
import glob
import os
import subprocess as sp
import sys
import tempfile

import pytest
from polygraphy.logger import G_LOGGER
from polygraphy.util import misc

import tensorrt as trt
from tests.common import check_file_non_empty, version
from tests.models.meta import ONNX_MODELS, TF_MODELS
from tests.tools.common import run_subtool, run_polygraphy_inspect, run_polygraphy_run


#
# INSPECT MODEL
#

@pytest.fixture(scope="module", params=["none", "basic", "attrs", "full"])
def run_inspect_model(request):
    yield lambda additional_opts: run_polygraphy_inspect(["model"] + ["--mode={:}".format(request.param)] + additional_opts)


@pytest.mark.parametrize("model", ["identity", "scan", "tensor_attr"])
def test_polygraphy_inspect_model_trt(run_inspect_model, model):
    if model == "tensor_attr" and version(trt.__version__) < version("7.2"):
        pytest.skip("Models with constant outputs were not supported before 7.2")

    if model == "scan" and version(trt.__version__) < version("7.0"):
        pytest.skip("Scan was not supported until 7.0")

    run_inspect_model([ONNX_MODELS[model].path, "--display-as=trt"])


def test_polygraphy_inspect_model_trt_engine_sanity(run_inspect_model):
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--trt", "--save-engine", outpath.name])
        run_inspect_model([outpath.name, "--model-type=engine"])


@pytest.mark.parametrize("model", ["identity", "scan", "tensor_attr"])
def test_polygraphy_inspect_model_onnx(run_inspect_model, model):
    run_inspect_model([ONNX_MODELS[model].path])


def test_polygraphy_inspect_model_tf_sanity(run_inspect_model):
    run_inspect_model([TF_MODELS["identity"].path, "--model-type=frozen"])


#
# INSPECT RESULTS
#

@pytest.mark.parametrize("opts", [[], ["--show-values"]])
def test_polygraphy_inspect_results(opts):
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outpath.name])
        run_polygraphy_inspect(["results", outpath.name] + opts)
