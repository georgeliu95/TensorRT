#!/usr/bin/env python3

"""
This script runs an identity model with ONNX-Runtime and TensorRT,
then compares outputs.
"""
from polygraphy.backend.trt import NetworkFromOnnxBytes, EngineFromNetwork, TrtRunner
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnxBytes
from polygraphy.backend.common import BytesFromPath
from polygraphy.comparator import Comparator

import os

# Create loaders for both ONNX Runtime and TensorRT
MODEL = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "models", "identity.onnx")

load_serialized_onnx = BytesFromPath(MODEL)
build_onnxrt_session = SessionFromOnnxBytes(load_serialized_onnx)
build_engine = EngineFromNetwork(NetworkFromOnnxBytes(load_serialized_onnx))

# Create runners
runners = [
    TrtRunner(build_engine),
    OnnxrtRunner(build_onnxrt_session),
]

# Finally, run and compare the results.
run_results = Comparator.run(runners)
assert bool(Comparator.compare_accuracy(run_results))
