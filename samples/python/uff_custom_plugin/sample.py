#
# Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

import sys
import os
import ctypes
from random import randint

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit

import tensorrt as trt

# ../common.py
sys.path.insert(1,
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir
    )
)
import common

# MNIST dataset metadata
MNIST_IMAGE_SIZE = 28
MNIST_CHANNELS = 1
MNIST_CLASSES = 10

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

# Path where clip plugin library will be built (check README.md)
CLIP_PLUGIN_LIBRARY = os.path.join(
    WORKING_DIR,
    'build/libclipplugin.so'
)

# Path to which trained model will be saved (check README.md)
MODEL_DIR = os.path.join(
    WORKING_DIR,
    'models'
)
MODEL_PATH = os.path.join(
    MODEL_DIR,
    'trained_lenet5.uff'
)

# Define global logger object (it should be a singleton,
# available for TensorRT from anywhere in code).
# You can set the logger severity higher to suppress messages
# (or lower to display more messages)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Define some global constants about the model.
class ModelData(object):
    INPUT_NAME = "InputLayer"
    INPUT_SHAPE = (MNIST_CHANNELS, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
    RELU6_NAME = "ReLU6"
    OUTPUT_NAME = "OutputLayer/Softmax"
    OUTPUT_SHAPE = (MNIST_IMAGE_SIZE, )
    DATA_TYPE = trt.float32

# Builds TensorRT Engine
def build_engine(model_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.UffParser() as parser, trt.Runtime(TRT_LOGGER) as runtime:
        config.max_workspace_size = common.GiB(1)

        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        parser.parse(model_path, network)

        plan = builder.build_serialized_network(network, config)
        return runtime.deserialize_cuda_engine(plan)

# Loads a test case into the provided pagelocked_buffer. Returns loaded test case label.
def load_normalized_test_case(pagelocked_buffer):
    x_test = np.load(os.path.join(MODEL_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(MODEL_DIR, "y_test.npy"))
    num_test = len(x_test)
    case_num = randint(0, num_test-1)
    img = x_test[case_num].ravel()
    np.copyto(pagelocked_buffer, img)
    return y_test[case_num]

def main():
    # Load the shared object file containing the Clip plugin implementation.
    # By doing this, you will also register the Clip plugin with the TensorRT
    # PluginRegistry through use of the macro REGISTER_TENSORRT_PLUGIN present
    # in the plugin implementation. Refer to plugin/clipPlugin.cpp for more details.
    if not os.path.isfile(CLIP_PLUGIN_LIBRARY):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load library ({}).".format(CLIP_PLUGIN_LIBRARY),
            "Please build the Clip sample plugin.",
            "For more information, see the included README.md"
        ))
    ctypes.CDLL(CLIP_PLUGIN_LIBRARY)

    # Load pretrained model
    if not os.path.isfile(MODEL_PATH):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load model file ({}).".format(MODEL_PATH),
            "Please use 'python lenet5.py' to train and save the model.",
            "For more information, see the included README.md"
        ))

    # Build an engine and retrieve the image mean from the model.
    with build_engine(MODEL_PATH) as engine:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        with engine.create_execution_context() as context:
            print("\n=== Testing ===")
            test_case = load_normalized_test_case(inputs[0].host)
            print("Loading Test Case: " + str(test_case))
            # The common do_inference function will return a list of outputs - we only have one in this case.
            [pred] = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            print("Prediction: " + str(np.argmax(pred)))


if __name__ == "__main__":
    main()
