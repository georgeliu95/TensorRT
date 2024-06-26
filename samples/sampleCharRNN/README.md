# Building An RNN Network Layer By Layer

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
	* [TensorRT API layers and ops](#tensorrt-api-layers-and-ops)
- [Preparing sample data](#preparing-sample-data)
- [Converting TensorFlow weights](#converting-tensorflow-weights)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample-help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, sampleCharRNN, uses the TensorRT API to build an RNN network layer by layer, sets up weights and inputs/outputs and then performs inference. Specifically, this sample creates a CharRNN network that has been trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset. For more information about character level modeling, see [char-rnn](https://github.com/karpathy/char-rnn).

TensorFlow has a useful  [RNN Tutorial](https://www.tensorflow.org/tutorials/recurrent)  which can be used to train a word level model. Word level models learn a probability distribution over a set of all possible word sequence. Since our goal is to train a char level model, which learns a probability distribution over a set of all possible characters, a few modifications will need to be made to get the TensorFlow sample to work. These modifications can be seen  [here](http://deeplearningathome.com/2016/10/Text-generation-using-deep-recurrent-neural-networks.html).

There are also many GitHub repositories that contain CharRNN implementations that will work out of the box. [Tensorflow-char-rnn](https://github.com/crazydonkey200/tensorflow-char-rnn)  is one such implementation.

## How does this sample work?

The CharRNN network is a fairly simple RNN network. The input into the network is a single character that is embedded into a vector of size 512. This embedded input is then supplied to a RNN layer containing two stacked LSTM cells. The output from the RNN layer is then supplied to a fully connected layer, which can be represented in TensorRT by a Matrix Multiply layer followed by an ElementWise sum layer. Constant layers are used to supply the weights and biases to the Matrix Multiply and ElementWise Layers, respectively. A TopK operation is then performed on the output of the ElementWise sum layer where `K = 1` to find the next predicted character in the sequence. For more information about these layers, see the [TensorRT API](http://docs.nvidia.com/deeplearning/sdk/tensorrt-api/index.html) documentation.

This sample provides a pre-trained model called `model-20080.data-00000-of-00001` located in the `/usr/src/tensorrt/data/samples/char-rnn/model` directory, therefore, training is not required for this sample. The model used by this sample was trained using [tensorflow-char-rnn](https://github.com/crazydonkey200/tensorflow-char-rnn). This GitHub repository includes instructions on how to train and produce checkpoint that can be used by TensorRT.

**Note:** If you wanted to train your own model and then perform inference with TensorRT, you will simply need to do a char to char comparison between TensorFlow and TensorRT.


### TensorRT API layers and ops

In this sample, the following layers are used.  For more information about these layers, see the [TensorRT Developer Guide: Layers](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#layers) documentation.

[ElementWise](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#elementwise-layer)
The ElementWise layer, also known as the Eltwise layer, implements per-element operations. The ElementWise layer is used to execute the second step of the functionality provided by a FullyConnected layer.

[MatrixMultiply](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#matrixmultiply-layer)
The MatrixMultiply layer implements matrix multiplication for a collection of matrices. The Matrix Multiplication layer is used to execute the first step of the functionality provided by a FullyConnected layer.

[TopK](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#topk-layer)
The TopK layer is used to identify the character that has the maximum probability of appearing next. The TopK layer finds the top K maximum (or minimum) elements along a dimension, returning a reduced tensor and a tensor of index positions.

## Preparing sample data

1. Download the sample data from [TensorRT release tarball](https://developer.nvidia.com/nvidia-tensorrt-download#), if not already mounted under `/usr/src/tensorrt/data` (NVIDIA NGC containers) and set it to `$TRT_DATADIR`.
    ```bash
    export TRT_DATADIR=/usr/src/tensorrt/data
    ```

## Converting TensorFlow weights

(Optional) If you want to train your own model and not use the pre-trained model included in this sample, you’ll need to convert the TensorFlow weights into a format that TensorRT can use.

1.  Locate TensorFlow weights dumping script:  
    ```bash
    $TRT_OSSPATH/samples/common/dumpTFWts.py
    ```

	This script has been provided to extract the weights from the model checkpoint files that are created during training. Use `dumpTFWts.py -h` for directions on the usage of the script.

2.  Convert the TensorFlow weights using the following command:
    ```bash
    dumpTFWts.py -m /path/to/checkpoint -o /path/to/output
    ```

## Running the sample

1. Compile the sample by following build instructions in [TensorRT README](https://github.com/NVIDIA/TensorRT/).

2.  Run the sample to generate characters based on the trained model:
    ```bash
    ./sample_char_rnn --datadir=<path/to/data>
    ```

    For example:
    ```bash
    ./sample_char_rnn --datadir $TRT_DATADIR/char-rnn
    ```

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	&&&& RUNNING TensorRT.sample_char_rnn # ./sample_char_rnn
	[I] [TRT] Detected 4 input and 3 output network tensors.
	[I] RNN Warmup: JACK
	[I] Expect: INGHAM:
	What shall I
	[I] Received: INGHAM:
	What shall I
	&&&& PASSED TensorRT.sample_char_rnn # ./sample_char_rnn
	```
	This output shows that the sample ran successfully; `PASSED`.

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option.


# Additional resources

The following resources provide a deeper understanding about RNN networks:

**RNN networks**
- [GNMT](https://arxiv.org/pdf/1609.08144v1.pdf)
- [NMT](https://arxiv.org/pdf/1701.02810.pdf)
- [Transformer](https://arxiv.org/pdf/1706.03762.pdf)

**Videos**
- [Introduction to RNNs in TensorRT](https://www.youtube.com/watch?reload=9&v=G3QA3ZzD4oc)

**Documentation**
- [TensorRT Sample Support Guide: sampleCharRNN](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#charRNN_sample)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) 
documentation.


# Changelog

January 2024
* Removed RNNv2Layer based addLSTMLayer implementation. addLSTMLayer is now implemented with ILoop only.
* Default to use ILoop in paramaters.

February 2019
This is the first release of this `README.md` file.


# Known issues

There are no known issues in this sample.
