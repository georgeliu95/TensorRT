# NVIDIA TensorRT Sample for Safety "SampleSafeMNIST"

**Table Of Contents**
- [NVIDIA TensorRT Sample for Safety "SampleSafeMNIST"](#NVIDIA-TensorRT-Sample-for-Safety-%22SampleSafeMNIST%22)
  - [Description](#Description)
  - [How does this sample work?](#How-does-this-sample-work)
    - [TensorRT API layers and ops](#TensorRT-API-layers-and-ops)
  - [Running the sample](#Running-the-sample)
    - [Sample --help options](#Sample---help-options)
- [Additional resources](#Additional-resources)
- [License](#License)
- [Changelog](#Changelog)
- [Known issues](#Known-issues)

## Description

Sample to demonstrate how to use the safe runtime, engine, execution context, and builder flag for safety.

## How does this sample work?

This sample uses a Caffe model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md).

Specifically, this sample:
- Performs the basic setup and initialization of TensorRT using the Caffe parser    
- [Imports a trained Caffe model using Caffe parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_caffe_c)    
- Preprocesses the input and stores the result in a managed buffer
- [Builds a safe engine](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c)
- [Serializes and deserializes the engine](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c)
- [Uses the engine to perform inference on an input image](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c)

To verify whether the engine is operating correctly, this sample picks a 28x28 image of a digit at random and runs inference on it using the engine it created. The output of the network is a probability distribution on the digit, showing which digit is likely that in the image.

### TensorRT API layers and ops

In this sample, the following layers are used.  For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[Activation layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#activation-layer)
The Activation layer implements element-wise activation functions. Specifically, this sample uses the Activation layer with the type `kRELU`. 

[Convolution layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#convolution-layer)
The Convolution layer computes a 2D (channel, height, and width) convolution, with or without bias.

[FullyConnected layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#fullyconnected-layer)
The FullyConnected layer implements a matrix-vector product, with or without bias.

[Pooling layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#pooling-layer)
The Pooling layer implements pooling within a channel. Supported pooling types are `maximum`, `average` and `maximum-average blend`.

[Scale layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#scale-layer)
The Scale layer implements a per-tensor, per-channel, or per-element affine transformation and/or exponentiation by constant values.

[SoftMax layer](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#softmax-layer)
The SoftMax layer applies the SoftMax function on the input tensor along an input dimension specified by the user.

## Running the sample

1. Compile the samples.
    ```
    cd <TensorRT root directory>/samples/sampleSafeMNIST
    mkdir build
    cd build/
    cmake ..
    make
    ```
    Where `<TensorRT root directory>` is where you installed TensorRT.

    This will generate two executable files, one for building safe engine, the other for inference.

    **Note:** If any of the dependencies are not installed in their default locations, you can manually specify them. Safe mode will be enabled in release build, or use -DTRT_SAFE_BUILD=ON to set it manually. For example:
    ```
    cmake .. -DCUDA_ROOT=/usr/local/cuda-10.2/
    -DNVINFER_LIB=/path/to/libnvinfer.so -DTRT_INC_DIR=/path/to/tensorrt/include/
    -DTRT_SAFE_BUILD=ON
    ```

2. Run the sample to build a TensorRT safe engine.
    ```
    ./sample_mnist_safe_build [--datadir=/path/to/data/dir/] [--fp16 or --int8]
    ```
    This sample reads three Caffe files to build the network:
    -   `mnist.prototxt` 
    The prototxt file that contains the network design.

    -   `mnist.caffemodel`
    The model file which contains the trained weights for the network.

    -   `mnist_mean.binaryproto`
    The binaryproto file which contains the means.

    This sample can be run in FP16 and INT8 modes as well.

    This sample builds a safe version TensorRT engine and saves it into a binary file.

    -   `safe_mnist.engine`
    The binary file which contains the serialized engine data.

    **Note:** By default, the sample expects these files to be in either the `data/samples/mnist/` or `data/mnist/` directories. The list of default directories can be changed by adding one or more paths with `--datadir=/new/path/` as a command line argument.

3. Verify the sample ran successfully. If the sample runs successfully you will see output similar to the following:
    ```
    &&&& RUNNING TensorRT.sample_mnist_safe_build # ./sample_mnist_safe_build
    [I] Building a GPU inference engine for MNIST
    [I] [TRT] Detected 1 input and 1 output network tensors.
    &&&& PASSED TensorRT.sample_mnist_safe_build # ./sample_mnist_safe_build
    ```

	This output shows that the sample ran successfully; `PASSED`.

4. Run the sample to perform inference on the digit:
    ```
    ./sample_mnist_safe_infer [-h] [--datadir=/path/to/data/dir/]
   
    This sample loads the prebuilt engine file.
    ```

5.  Verify that the sample ran successfully. If the sample runs successfully you will see output similar to the following; ASCII rendering of the input image with digit 3:
    ```
    &&&& RUNNING TensorRT.sample_mnist_safe_infer # ./sample_mnist_safe_infer
    [I] Running a GPU inference engine for MNIST
    [I] Input:
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@#-:.-=@@@@@@@@@@@@@@
    @@@@@%=     . *@@@@@@@@@@@@@
    @@@@% .:+%%%  *@@@@@@@@@@@@@
    @@@@+=#@@@@@# @@@@@@@@@@@@@@
    @@@@@@@@@@@%  @@@@@@@@@@@@@@
    @@@@@@@@@@@: *@@@@@@@@@@@@@@
    @@@@@@@@@@- .@@@@@@@@@@@@@@@
    @@@@@@@@@:  #@@@@@@@@@@@@@@@
    @@@@@@@@:   +*%#@@@@@@@@@@@@
    @@@@@@@%         :+*@@@@@@@@
    @@@@@@@@#*+--.::     +@@@@@@
    @@@@@@@@@@@@@@@@#=:.  +@@@@@
    @@@@@@@@@@@@@@@@@@@@  .@@@@@
    @@@@@@@@@@@@@@@@@@@@#. #@@@@
    @@@@@@@@@@@@@@@@@@@@#  @@@@@
    @@@@@@@@@%@@@@@@@@@@- +@@@@@
    @@@@@@@@#-@@@@@@@@*. =@@@@@@
    @@@@@@@@ .+%%%%+=.  =@@@@@@@
    @@@@@@@@           =@@@@@@@@
    @@@@@@@@*=:   :--*@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@

    [I] Output:
    0:
    1:
    2:
    3: **********
    4:
    5:
    6:
    7:
    8:
    9:

    &&&& PASSED TensorRT.sample_safe_mnist_infer # ./sample_mnist_safe_infer
    ```

    This output shows that the sample ran successfully; `PASSED`.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
Usage: ./sample_mnist_safe_build [-h or --help] [-d or --datadir=<path to data directory>]
--help Display help information
--datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
--int8 Run in Int8 mode.
--fp16 Run in FP16 mode.

Usage: ./sample_mnist_safe_infer [-h or --help] [-d or --datadir=<path to data directory>]
--help Display help information
--datadir Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)
```

# Additional resources

The following resources provide a deeper understanding about sampleSafeMNIST:

**MNIST**
- [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

June 2019
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
