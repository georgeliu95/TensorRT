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

from PIL import Image
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import graphsurgeon as gs
import uff

# lenet5.py
import lenet5


# MNIST dataset metadata
MNIST_IMAGE_SIZE = 28
MNIST_CHANNELS = 1
MNIST_CLASSES = 10

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(os.path.realpath(__file__))

# Path to which trained model will be saved (check README.md)
MODEL_PATH = os.path.join(
    WORKING_DIR,
    'models/trained_lenet5.pb'
)

# Define some global constants about the model.
class ModelData(object):
    INPUT_NAME = "InputLayer"
    INPUT_SHAPE = (MNIST_CHANNELS, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
    RELU6_NAME = "ReLU6"
    OUTPUT_NAME = "OutputLayer/Softmax"
    OUTPUT_SHAPE = (MNIST_IMAGE_SIZE, )


# Generates mappings from unsupported TensorFlow operations to TensorRT plugins
def prepare_namespace_plugin_map():
    # In this sample, the only operation that is not supported by TensorRT
    # is tf.nn.relu6, so we create a new node which will tell UffParser which
    # plugin to run and with which arguments in place of tf.nn.relu6.


    # The "clipMin" and "clipMax" fields of this TensorFlow node will be parsed by createPlugin,
    # and used to create a CustomClipPlugin with the appropriate parameters.
    trt_relu6 = gs.create_plugin_node(name="trt_relu6", op="CustomClipPlugin", clipMin=0.0, clipMax=6.0)
    namespace_plugin_map = {
        ModelData.RELU6_NAME: trt_relu6
    }
    return namespace_plugin_map

# Transforms model path to uff path (e.g. /a/b/c/d.pb -> /a/b/c/d.uff)
def model_path_to_uff_path(model_path):
    uff_path = os.path.splitext(model_path)[0] + ".uff"
    return uff_path

# Converts the TensorFlow frozen graphdef to UFF format using the UFF converter
def model_to_uff(model_path):
    # Transform graph using graphsurgeon to map unsupported TensorFlow
    # operations to appropriate TensorRT custom layer plugins
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph.collapse_namespaces(prepare_namespace_plugin_map())
    # Save resulting graph to UFF file
    output_uff_path = model_path_to_uff_path(model_path)
    uff.from_tensorflow(
        dynamic_graph.as_graph_def(),
        [ModelData.OUTPUT_NAME],
        output_filename=output_uff_path,
        text=True
    )
    return output_uff_path

def main():
    # Load pretrained model
    if not os.path.isfile(MODEL_PATH):
        raise IOError("\n{}\n{}\n{}\n".format(
            "Failed to load model file ({}).".format(MODEL_PATH),
            "Please use 'python lenet5.py' to train and save the model.",
            "For more information, see the included README.md"
        ))

    uff_path = model_to_uff(MODEL_PATH)

if __name__ == "__main__":
    main()
