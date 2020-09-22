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
from tests.tools.common import run_subtool, run_polygraphy_precision, run_polygraphy_run


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_polygraphy_precision_bisect_sanity():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outpath.name])
        run_polygraphy_precision(["bisect", ONNX_MODELS["identity"].path, "--golden", outpath.name, "--int8"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_polygraphy_precision_linear_sanity():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outpath.name])
        run_polygraphy_precision(["linear", ONNX_MODELS["identity"].path, "--golden", outpath.name, "--int8"])


@pytest.mark.skipif(version(trt.__version__) < version("7.0"), reason="Unsupported for TRT 6")
def test_polygraphy_precision_worst_first_sanity():
    with tempfile.NamedTemporaryFile() as outpath:
        run_polygraphy_run([ONNX_MODELS["identity"].path, "--onnxrt", "--save-results", outpath.name, "--onnx-outputs", "mark", "all"])
        run_polygraphy_precision(["worst-first", ONNX_MODELS["identity"].path, "--golden", outpath.name, "--int8", "--trt-outputs", "mark", "all"])
