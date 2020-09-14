#!/usr/bin/env python3

"""
This script uses the Polygraphy Runner API to validate the outputs
of an identity model using a trivial dataset.
"""
from polygraphy.backend.trt import NetworkFromOnnxPath, EngineFromNetwork, TrtRunner

import numpy as np
import os


INPUT_SHAPE = (1, 1, 2, 2)
REAL_DATASET = [ # Definitely real data
    np.ones(INPUT_SHAPE, dtype=np.float32),
    np.zeros(INPUT_SHAPE, dtype=np.float32),
    np.ones(INPUT_SHAPE, dtype=np.float32),
    np.zeros(INPUT_SHAPE, dtype=np.float32),
]

# For our identity network, the golden output values are the same as the input values.
# Though this network appears to do nothing, it can be incredibly useful in some cases (like here!).
GOLDEN_VALUES = REAL_DATASET

MODEL = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "models", "identity.onnx")

build_engine = EngineFromNetwork(NetworkFromOnnxPath(MODEL))

# Activate the runner using a context manager. For TensorRT, this will build an engine,
# then destroy it upon exiting the context.
# NOTE: You can also use the activate() function for this, but you will need to make sure to
# deactivate() to avoid a memory leak. For that reason, a context manager is the safer option.
with TrtRunner(build_engine) as runner:
    for (data, golden) in zip(REAL_DATASET, GOLDEN_VALUES):
        outputs = runner.infer(feed_dict={"x": data})
        assert np.all(outputs["y"] == golden)
