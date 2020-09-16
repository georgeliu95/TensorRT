#!/usr/bin/env python3

"""
This script demonstrates how to use the extend() API covered in example 03
to construct a TensorRT network using the TensorRT Network API.
"""
from polygraphy.backend.trt import CreateNetwork, EngineFromNetwork, TrtRunner
from polygraphy.common.func import extend

import tensorrt as trt
import numpy as np


INPUT_NAME = "input"
INPUT_SHAPE = (64, 64)
OUTPUT_NAME = "output"

# Just like in example 03, we can use `extend` to add our own functionality to existing loaders.
# `CreateNetwork` will create an empty network, which we can then populate ourselves.
@extend(CreateNetwork())
def create_network(builder, network):
    # This network will add 1 to the input tensor
    inp = network.add_input(name=INPUT_NAME, shape=INPUT_SHAPE, dtype=trt.float32)
    ones = network.add_constant(shape=INPUT_SHAPE, weights=np.ones(shape=INPUT_SHAPE, dtype=np.float32)).get_output(0)
    add = network.add_elementwise(inp, ones, op=trt.ElementWiseOperation.SUM).get_output(0)
    add.name = OUTPUT_NAME
    network.mark_output(add)


# After we've constructed the network, we can go back to using regular Polygraphy APIs.
build_engine = EngineFromNetwork(create_network)

with TrtRunner(build_engine) as runner:
    feed_dict = {INPUT_NAME: np.random.random_sample(INPUT_SHAPE).astype(np.float32)}
    outputs = runner.infer(feed_dict)
    assert np.all(outputs[OUTPUT_NAME] == (feed_dict[INPUT_NAME] + 1))