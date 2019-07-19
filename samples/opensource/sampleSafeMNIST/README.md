# “Hello World” For TensorRT Safety


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleSafeMNIST, consists of two parts; build and infer. The build part of this sample demonstrates how to use the builder `IBuilderConfig::setEngineCapability()` flag for safety. The inference part of this sample demonstrates how to use the safe runtime, engine and execution context.

The build part builds a safe version of a TensorRT engine and saves it into a binary file, then the infer part loads the prebuilt safe engine and performs inference on an input image. The infer part uses the safety header proxy, with the `CMakeLists.txt` file demonstrating how to build it against the safe runtime for deployment and extended runtime for development. This sample can be run in FP16 and INT8 modes.

## How does this sample work?

This sample uses a Caffe model that was trained on the [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md).

Specifically, this sample:
-   Performs the basic setup and initialization of TensorRT using the Caffe parser
-   [Imports a trained Caffe model using Caffe parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#import_caffe_c)
-   Preprocesses the input and stores the result in a managed buffer
-   [Builds a safe engine](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#build_engine_c)
-   [Serializes and deserializes the engine](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#serial_model_c)
-   [Uses the engine to perform inference on an input image](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c)

To verify whether the engine is operating correctly, this sample picks a 28x28 image of a digit at random and runs inference on it using the engine it created. The output of the network is a probability distribution on the digit, showing which digit is likely that in the image.

### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

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

1. Download the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to read images from the ubyte file. The images need to be saved into `.pgm` format and renamed as `<label>.pgm`.

2. Download the [Caffe prototxt file](https://github.com/BVLC/caffe/tree/master/examples/mnist) to prepare the `caffemodel` and `mean` file for LeNet-5.

3. Put the images and the Caffe files into the `/data/samples/mnist` directory.

4. Use `cmake` to generate the `Makefile` for this sample.  Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleSafeMNIST/build` directory. The binary named `sample_mnist_safe_build` and `sample_mnist_safe_infer` will be created in the current directory.
	```
	cd <TensorRT root directory>/samples/sampleSafeMNIST
	mkdir build
	cd build/
	cmake ..
	make
	```
 
	Where `<TensorRT root directory>` is where you installed TensorRT.

	**Note:** If any of the dependencies are not installed in their default locations, you can manually specify them. Set `-DTRT_SAFE_BUILD=ON` to enable safe mode for the infer part build, which means building against safe runtime. Otherwise, it will link against the extended runtime lib. Safe mode is forced to be enabled in the release build. For example:
	```
	cmake .. -DCUDA_ROOT=/usr/local/cuda-10.2/ -DTRT_LIB_DIR=/path/to/tensorrt/libs -DTRT_INC_DIR=/path/to/tensorrt/include/ -DCMAKE_BUILD_TYPE=RELEASE
	```
	Or
	```
	cmake .. -DCUDA_ROOT=/usr/local/cuda-10.2/ -DTRT_LIB_DIR=/path/to/tensorrt/libs -DTRT_INC_DIR=/path/to/tensorrt/include/ -DTRT_SAFE_BUILD=ON
	```

5.  Run the sample to build a TensorRT safe engine.
	```
	./sample_mnist_safe_build [--datadir=/path/to/data/dir/] [--fp16 or --int8]
	```
	
	This sample generates `safe_mnist.engine`, which is a binary file that contains the serialized engine data.

	This sample reads three Caffe files to build the network:
	- `mnist.prototxt` - The prototxt file that contains the network design.
	- `mnist.caffemodel` - The model file which contains the trained weights for the network.
	- `mnist_mean.binaryproto` - The binaryproto file which contains the means.
	  
	**Note:** By default, this sample expects these files to be in either the `data/samples/mnist/` or `data/mnist/` directories. The list of default directories can be changed by adding one or more paths with `--datadir=/new/path/` as a command line argument.

6. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_mnist_safe_build # ./sample_mnist_safe_build
	[I] Building a GPU inference engine for MNIST
	[I] [TRT] Detected 1 input and 1 output network tensors.
	&&&& PASSED TensorRT.sample_mnist_safe_build # ./sample_mnist_safe_build
	```
	This output shows that the sample ran successfully; `PASSED`.

7. Run the sample to perform inference on the digit:
	`./sample_mnist_safe_infer`

	**Note:** This sample expects `./sample_mnist_safe_build` has been run to generate a safe engine file. It loads input image from `data/sample/mnist` directory, and walks back 10 directories to locate the image.

8. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following; ASCII rendering of the input image with digit 3:
	```
	&&&& RUNNING TensorRT.sample_mnist_safe_infer # ./sample_mnist_safe_infer
	[I] Running a GPU inference engine for MNIST
	[I] Input:
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@#-:.-=@@@@@@@@@@@@@@
	@@@@@%= . *@@@@@@@@@@@@@@@@@
	@@@@% .:+%%% *@@@@@@@@@@@@@@
	@@@@+=#@@@@@# @@@@@@@@@@@@@@
	@@@@@@@@@@@% @@@@@@@@@@@@@@@
	@@@@@@@@@@@: *@@@@@@@@@@@@@@
	@@@@@@@@@@- .@@@@@@@@@@@@@@@
	@@@@@@@@@: #@@@@@@@@@@@@@@@@
	@@@@@@@@: +*%#@@@@@@@@@@@@@@
	@@@@@@@% :+*@@@@@@@@@@@@@@@@
	@@@@@@@@#*+--.:: +@@@@@@@@@@
	@@@@@@@@@@@@@@@@#=:. +@@@@@@
	@@@@@@@@@@@@@@@@@@@@ .@@@@@@
	@@@@@@@@@@@@@@@@@@@@#. #@@@@
	@@@@@@@@@@@@@@@@@@@@# @@@@@@
	@@@@@@@@@%@@@@@@@@@@- +@@@@@
	@@@@@@@@#-@@@@@@@@*. =@@@@@@
	@@@@@@@@ .+%%%%+=. =@@@@@@@@
	@@@@@@@@ =@@@@@@@@@@@@@@@@@@
	@@@@@@@@*=: :--*@@@@@@@@@@@@
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


### Sample `--help` options

To see the full list of available options and their descriptions, use the `./sample_mnist_safe_build [-h or --help]` command.

## Additional resources

The following resources provide a deeper understanding about sampleSafeMNIST.

**Dataset**
- [MNIST dataset](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)

**Documentation**
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Changelog

June 2019
This is the first release of the `README.md` file and sample.


## Known issues

There are no known issues in this sample.