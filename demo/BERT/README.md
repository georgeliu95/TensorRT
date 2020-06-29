# BERT Inference Using TensorRT

This subfolder of the BERT TensorFlow repository, tested and maintained by NVIDIA, provides scripts to perform high-performance inference using NVIDIA TensorRT.


## Table Of Contents

- [Model Overview](#model-overview)
   * [Model Architecture](#model-architecture)
   * [TensorRT Inference Pipeline](#tensorrt-inference-pipeline)
   * [Version Info](#version-info)
- [Setup](#setup)
   * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
  * [(Optional) Trying a different configuration](#optional-trying-a-different-configuration)
- [Advanced](#advanced)
   * [Scripts and sample code](#scripts-and-sample-code)
   * [Command-line options](#command-line-options)
   * [TensorRT inference process](#tensorrt-inference-process)
- [Performance](#performance)
  * [Benchmarking](#benchmarking)
       * [TensorRT inference benchmark](#tensorrt-inference-benchmark)


## Model overview

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. This model is based on the [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) paper. NVIDIA's BERT is an optimized version of [Google's official implementation](https://github.com/google-research/bert), leveraging mixed precision arithmetic and Tensor Cores for faster inference times while maintaining target accuracy.

Other publicly available implementations of BERT include:
1. [NVIDIA PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT)
2. [Hugging Face](https://github.com/huggingface/pytorch-pretrained-BERT)
3. [codertimo](https://github.com/codertimo/BERT-pytorch)
4. [gluon-nlp](https://github.com/dmlc/gluon-nlp/tree/master/scripts/bert)
5. [Google's official implementation](https://github.com/google-research/bert)


### Model architecture

BERT's model architecture is a multi-layer bidirectional Transformer encoder. Based on the model size, we have the following two default configurations of BERT:

| **Model** | **Hidden layers** | **Hidden unit size** | **Attention heads** | **Feed-forward filter size** | **Max sequence length** | **Parameters** |
|:---------:|:----------:|:----:|:---:|:--------:|:---:|:----:|
|BERT-Base |12 encoder| 768| 12|4 x  768|512|110M|
|BERT-Large|24 encoder|1024| 16|4 x 1024|512|330M|

Typically, the language model is followed by a few task-specific layers. The model used here includes layers for question answering.

### TensorRT Inference Pipeline

BERT inference consists of three main stages: tokenization, the BERT model, and finally a projection of the tokenized prediction onto the original text.
Since the tokenizer and projection of the final predictions are not nearly as compute-heavy as the model itself, we run them on the host. The BERT model is GPU-accelerated via TensorRT.

The tokenizer splits the input text into tokens that can be consumed by the model. For details on this process, see [this tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/).

To run the BERT model in TensorRT, we construct the model using TensorRT APIs and import the weights from a pre-trained TensorFlow checkpoint from [NGC](https://ngc.nvidia.com/models/nvidian:bert_tf_v2_large_fp16_128). Finally, a TensorRT engine is generated and serialized to the disk. The various inference scripts then load this engine for inference.

Lastly, the tokens predicted by the model are projected back to the original text to get a final result.

### Version Info

The following software version configuration has been tested:

|Software|Version|
|--------|-------|
|Python|3.6.9|
|TensorRT|7.1.3.4|
|CUDA|11.0.171|


## Setup

The following section lists the requirements that you need to meet in order to run the BERT model.

### Requirements

This demo BERT application can be run within the TensorRT Open Source build container. If running in a different environment, ensure you have the following packages installed.

* [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) - for downloading BERT checkpoints from NGC.
* PyPI Packages:
  * [pycuda](https://pypi.org/project/pycuda/) 2019.1.2
  * [onnx](https://pypi.org/project/onnx/1.6.0/) 1.6.0
  * [tensorflow](https://pypi.org/project/tensorflow/1.15.3/) 1.15
* NVIDIA [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/), [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU with NVIDIA Driver 450.37 or later.


## Quick Start Guide

1. Build and launch the TensorRT-OSS build container. On x86 with Ubuntu 18.04 for example:
    ```bash
    cd <TensorRT-OSS>
    ./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu --os 18.04 --cuda 11.0
    ./docker/launch.sh --tag tensorrt-ubuntu --gpus all --release $TRT_RELEASE --source $TRT_SOURCE
    ```

    **Note:** After this point, all commands should be run from within the container.

2. Build the TensorRT Plugins library from source and install the TensorRT python bindings:
   ```bash
   cd $TRT_SOURCE
   export LD_LIBRARY_PATH=`pwd`/build/out:$LD_LIBRARY_PATH:/tensorrt/lib
   mkdir -p build && cd build
   cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out
   make -j$(nproc)

   pip3 install /tensorrt/python/tensorrt-7.1*-cp36-none-linux_x86_64.whl
   ```

3. Download the SQuAD dataset and BERT checkpoints:
    ```bash
    cd $TRT_SOURCE/demo/BERT
    ```

    Download SQuAD v1.1 training and dev dataset.
    ```bash
    bash scripts/download_squad.sh
    ```

    Download Tensorflow checkpoints for BERT large model with sequence length 128 and fp16 weights, fine-tuned for SQuAD v2.0.
    ```bash
    bash scripts/download_model.sh
    ````

**Note:** Since the datasets and checkpoints are stored in the directory mounted from the host, they do *not* need to be downloaded each time the container is launched. 

4. Build a TensorRT engine. To build an engine, run the `builder.py` script. For example:
    ```bash
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/model.ckpt-8144 -o /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine -b 1 -s 128 --fp16 -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), and sequence length of 128 (`-s 128`) using mixed precision (`--fp16`) using the BERT Large V2 FP16 Sequence Length 128 checkpoint (`-c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2`).

5. Run inference. Two options are provided for running the model.

    a. `inference.py` script
    This script accepts a passage and question and then runs the engine to generate an answer.
    For example:
    ```bash
    python3 inference.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/vocab.txt
    ```

    b. `inference.ipynb` Jupyter Notebook
    The Jupyter Notebook includes a passage and various example questions and allows you to interactively make modifications and see the outcome.
    To launch the Jupyter Notebook from inside the container, run:
    ```bash
    jupyter notebook --ip 0.0.0.0 inference.ipynb
    ```
    Then, use your browser to open the link displayed. The link should look similar to: `http://127.0.0.1:8888/?token=<TOKEN>`
    
6. Run inference with CUDA Graph support.

    A separate python `inference_c.py` script is provided to run inference with CUDA Graph support. This is necessary since CUDA Graph is only supported through CUDA C/C++ APIs, not pyCUDA. The `inference_c.py` script uses pybind11 to interface with C/C++ for CUDA graph capturing and launching. The cmdline interface is the same as `inference.py` except for an extra `--enable-graph` option.
    
    ```bash
    mkdir -p build
    cd build; cmake ..
    make; cd ..
    python3 inference_c.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine --enable-graph -p "TensorRT is a high performance deep learning inference platform that delivers low latency and high throughput for apps such as recommenders, speech and image/video on NVIDIA GPUs. It includes parsers to import models, and plugins to support novel ops and layers before applying optimizations for inference. Today NVIDIA is open-sourcing parsers and plugins in TensorRT so that the deep learning community can customize and extend these components to take advantage of powerful TensorRT optimizations for your apps." -q "What is TensorRT?" -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_128_v2/vocab.txt
    ```

    A separate C/C++ inference benchmark executable `perf` (compiled from `perf.cpp`) is provided to run inference benchmarks with CUDA Graph. The cmdline interface is the same as `perf.py` except for an extra `--enable_graph` option.
    
    ```bash
    build/perf -e /workspace/TensorRT/demo/BERT/engines/bert_large_128.engine -b 1 -s 128 -w 100 -i 1000 --enable_graph
    ```
    

### (Optional) Trying a different configuration

If you would like to run another configuration, you can manually download checkpoints using the included script. For example, run:
```bash
bash scripts/download_model.sh base
```
to download a BERT Base model instead of the default BERT Large model.

To view all available model options, run:
```bash
bash scripts/download_model.sh -h
```

## Advanced

The following sections provide greater details on inference with TensorRT.

### Scripts and sample code

In the `root` directory, the most important files are:

- `builder.py` - Builds an engine for the specified BERT model
- `Dockerfile` - Container which includes dependencies and model checkpoints to run BERT
- `inference.ipynb` - Runs inference interactively
- `inference.py` - Runs inference with a given passage and question
- `perf.py` - Runs inference benchmarks

The `scripts/` folder encapsulates all the one-click scripts required for running various supported functionalities, such as:

- `build.sh` - Builds a Docker container that is ready to run BERT
- `launch.sh` - Launches the container created by the `build.sh` script.
- `download_model.sh` - Downloads pre-trained model checkpoints from NGC
- `inference_benchmark.sh` - Runs an inference benchmark and prints results

Other folders included in the `root` directory are:

- `helpers` - Contains helpers for tokenization of inputs

The `infer_c/` folder contains all the necessary C/C++ files required for CUDA Graph support.
- `bert_infer.h` - Defines necessary data structures for running BERT inference
- `infer_c.cpp` - Defines C/C++ interface using pybind11 that can be plugged into `inference_c.py`
- `perf.cpp` - Runs inference benchmarks. It is equivalent to `perf.py`, with an extra option `--enable_graph` to enable CUDA Graph support.

### Command-line options

To view the available parameters for each script, you can use the help flag (`-h`).

### TensorRT inference process

As mentioned in the [Quick Start Guide](#quick-start-guide), two options are provided for running inference:
1. The `inference.py` script which accepts a passage and a question and then runs the engine to generate an answer. Alternatively, this script can be used to run inference on the Squad dataset.
2. The `inference.ipynb` Jupyter Notebook which includes a passage and various example questions and allows you to interactively make modifications and see the outcome.

## Accuracy

### Evaluating PTQ (post-training quantization) Int8 Accuracy Using The SQuAD Dataset
1.  Download Tensorflow checkpoints for a BERT Large FP16 SQuAD v2 model with a sequence length of 384:
    ```bash
    bash scripts/download_model.sh large fp16 384 v2
    ```

2. Build an engine:

    **Turing and Ampere GPUs**
    ```bash
    # QKVToContextPlugin and SkipLayerNormPlugin supported with INT8 I/O. To enable, use -imh and -iln builder flags respectively.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/model.ckpt-8144 -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 --squad-json ./squad/train-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt --calib-num 100 -iln -imh
    ```

    **Xavier GPU**
    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/model.ckpt-8144 -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 --squad-json ./squad/train-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt --calib-num 100 -iln 
    ```

    **Volta GPU**
    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -m /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/model.ckpt-8144 -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 --squad-json ./squad/train-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt --calib-num 100
    ```

    This will build an engine with a maximum batch size of 1 (`-b 1`), calibration dataset squad (`--squad-json ./squad/train-v1.1.json`), calibration sentences number 100 (`--calib-num 100`), and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:
    ```bash
    python3 inference.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```
### Evaluating QAT (quantization aware training) Int8 Accuracy Using The SQuAD Dataset
1.  Download checkpoint for BERT Large FP16 SQuAD v1.1 model with sequence length of 384:
    ```bash
    bash scripts/download_model.sh pyt v1_1
    ```

2. Build an engine:

    **Turing and Ampere GPUs**
    ```bash
    # QKVToContextPlugin and SkipLayerNormPlugin supported with INT8 I/O. To enable, use -imh and -iln builder flags respectively.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln -imh
    ```

    **Xavier GPU**
    ```bash
    # Only supports SkipLayerNormPlugin running with INT8 I/O. Use -iln builder flag to enable.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx -iln 
    ```

    **Volta GPU**
    ```bash
    # No support for QKVToContextPlugin or SkipLayerNormPlugin running with INT8 I/O. Don't specify -imh or -iln in builder flags.
    mkdir -p /workspace/TensorRT/demo/BERT/engines && python3 builder.py -o /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -b 1 -s 384 --int8 --fp16 --strict -c /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2 -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -x /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_pyt_onnx_large_qa_squad11_amp_fake_quant_v1/bert_large_v1_1_fake_quant.onnx 
    ```

    This will build and engine with a maximum batch size of 1 (`-b 1`) and sequence length of 384 (`-s 384`) using INT8 mixed precision computation where possible (`--int8 --fp16 --strict`).

3. Run inference using the squad dataset, and evaluate the F1 score and exact match score:

    ```bash
    python3 inference.py -e /workspace/TensorRT/demo/BERT/engines/bert_large_384_int8mix.engine -s 384 -sq ./squad/dev-v1.1.json -v /workspace/TensorRT/demo/BERT/models/fine-tuned/bert_tf_v2_large_fp16_384_v2/vocab.txt -o ./predictions.json
    python3 squad/evaluate-v1.1.py  squad/dev-v1.1.json  ./predictions.json 90
    ```

## Performance

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in inference modes.

#### TensorRT inference benchmark

The inference benchmark is performed on a single GPU by the `inference_benchmark.sh` script, which takes the following steps for each set of model parameters:

1. Downloads checkpoints and builds a TensorRT engine if it does not already exist.

2. Runs 1 warm-up iteration then runs inference for 100 iterations for each batch size specified in the script, selecting the profile best for each size.

**Note:** The time measurements do not include the time required to copy inputs to the device and copy outputs to the host.

To run the inference benchmark script, run:
```bash
bash scripts/inference_benchmark.sh
```

Note: Some of the configurations in the benchmark script require 16GB of GPU memory. On GPUs with smaller amounts of memory, parts of the benchmark may fail to run.

Also note that BERT Large engines, especially using mixed precision with large batch sizes and sequence lengths may take a couple hours to build.
