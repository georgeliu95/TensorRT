# Using The Nvmedia API To Run A TensorRT Engine


**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
   * [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
   * [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleNvmedia, uses an API to construct a network of a single ElementWise layer and builds the engine. The engine runs in DLA safe mode using Nvmedia runtime. In order to do that, the sample uses Nvmedia APIs to do engine conversion and Nvmedia runtime preparation, as well as the inference.

## How does this sample work?

Specifically:
-   The single-layered network is built by TensorRT.
-   `NvMediaDlaCreate` and `NvMediaDeviceCreate` are called to create DLA device.
-   `NvMediaDlaLoadFromMemory` is called to load the engine memory for DLA use.
-   `NvMediaTensorCreate` is called to create `NvMediaTensor`.
-   `NvMediaDlaSubmitTimeout` is called to submit the inference task.


### TensorRT API layers and ops

In this sample, the following layers are used. For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[ElementWise](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#elementwise-layer)
The ElementWise layer, also known as the Eltwise layer, implements per-element operations. The ElementWise layer is used to execute the second step of the functionality provided by a FullyConnected layer.

## Prerequisites

This sample needs to be compiled with macro `ENABLE_DLA=1`, otherwise, this sample will print the following error message:
```
Unsupported platform, please make sure it is running on aarch64, QNX or android.
```
and quit.

## Running the sample

1.  Compile this sample by running `make` in the `<TensorRT root directory>/samples/sampleNvmedia` directory. The binary named `sample_nvmedia` will be created in the `<TensorRT root directory>/bin` directory.
	```
	cd <TensorRT root directory>/samples/sampleNvmedia
	make
	```
 
	Where `<TensorRT root directory>` is where you installed TensorRT.

2.  Run the sample to perform inference on DLA.
    `./samplenvmedia`

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_nvmedia # ./sample_nvmedia
	[I] [TRT]
	[I] [TRT] --------------- Layers running on DLA:
	[I] [TRT] {(Unnamed Layer* 0) [ElementWise]},
	[I] [TRT] --------------- Layers running on GPU:
	[I] [TRT]
	…(omit messages)
	&&&& PASSED TensorRT.sample_nvmedia
	```
	  
	This output shows that the sample ran successfully; `PASSED`.


### Sample `--help` options

To see the full list of available options and their descriptions, use the `./sample_nvmedia -h` command line option. 


## Additional resources

The following resources provide a deeper understanding of sampleNvmedia.

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

## License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


## Changelog

July 2019
This is the first release of the `README.md` file.


## Known issues

There are no known issues with this tool.
