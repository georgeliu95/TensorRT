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
from tests.tools.common import (run_polygraphy_run, run_polygraphy_surgeon,
                                run_subtool)


def test_polygraphy_surgeon_sanity():
    with tempfile.NamedTemporaryFile() as configpath, tempfile.NamedTemporaryFile() as modelpath:
        run_polygraphy_surgeon(["prepare", ONNX_MODELS["identity"].path, "-o", configpath.name])
        run_polygraphy_surgeon(["operate", ONNX_MODELS["identity"].path, "-c", configpath.name, "-o", modelpath.name])
        run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])


def test_polygraphy_surgeon_extract_sanity():
    with tempfile.NamedTemporaryFile() as modelpath:
        run_polygraphy_surgeon(["extract", ONNX_MODELS["identity_identity"].path, "-o", modelpath.name, "--inputs", "identity_out_0,auto,auto"])
        run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])


def test_polygraphy_surgeon_extract_fallback_shape_inference():
    with tempfile.NamedTemporaryFile() as modelpath:
        # Force fallback shape inference by disabling ONNX shape inference
        run_polygraphy_surgeon(["extract", ONNX_MODELS["identity_identity"].path, "-o", modelpath.name, "--inputs",
                             "identity_out_0,auto,auto", "--outputs", "identity_out_2,auto", "--no-shape-inference"])
        run_polygraphy_run([modelpath.name, "--model-type=onnx", "--onnxrt"])
