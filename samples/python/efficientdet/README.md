# EfficientDet in TensorRT

These scripts help with conversion and execution of [Google EfficientDet](https://github.com/google/automl/tree/master/efficientdet) models with [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt).

## Setup

Install TensorRT as per the [TensorRT Install Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). You will need to make sure the Python bindings for TensorRT are also installed correctly, these are available by installing the `python3-libnvinfer` and `python3-libnvinfer-dev` packages on your TensorRT download.

Make sure all other packages listed in `requirements.txt` are also installed:

```
cat requirements.txt | xargs -n 1 -L 1 pip install
```

You will also want to clone the EfficientDet code from `https://github.com/google/automl` to use some helper utilities from it.

## Model Conversion

### TensorFlow Saved Model

The starting point of conversion is a TensorFlow training checkpoint, such as from your own trained models.

If you don't have a checkpoint yet, you can download one of the pre-trained checkpoints as described in the AutoML repository, such as:

```
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d0.tar.gz
```

Unpack the tar package to a directory, the resulting `eficientdet-d0` directory will contain the checkpoint files of the model.

You will then need to export a TensorFlow saved model from this. To do so, on the AutoML repository run:

```
cd /path/to/automl/efficientdet
python model_inspect.py \
    --runmode saved_model \
    --ckpt_path /path/to/checkpoint \
    --saved_model_dir /path/to/saved_model
```

This will create a directory named `saved_model` with the protobuf graph and other related files inside.

### Create ONNX Graph

To generate an ONNX model file, first find the input shape that corresponds to the model you're converting:

| **Model**        | **Input Shape** |
| -----------------|-----------------|
| EfficientDet D0  | N,512,512,3     |
| EfficientDet D1  | N,640,640,3     |
| EfficientDet D2  | N,768,768,3     |
| EfficientDet D3  | N,896,896,3     |
| EfficientDet D4  | N,1024,1024,3   |
| EfficientDet D5  | N,1280,1280,3   |
| EfficientDet D6  | N,1280,1280,3   |
| EfficientDet D7  | N,1536,1536,3   |
| EfficientDet D7x | N,1536,1536,3   |

Where **N** is the batch size you would like to run inference at, such as `8,512,512,3` for a batch size of 8.

The conversion process supports both NHWC and NCHW input formats, so if your input source is an `NCHW` data format, you can use the corresponding input shape, i.e. `1,512,512,3` -> `1,3,512,512`.

With the correct input shape selected, run:

```
python create_onnx.py \
    --saved_model /path/to/saved_model \
    --onnx /path/to/model.onnx \
    --input_shape '1,512,512,3'
```

This will create the file `model.onnx` which is ready to convert to TensorRT. 

You can visualize the resulting file with a tool such as [Netron](https://netron.app/).

The script has a few additional arguments:

* `--nms_threshold` allows overriding the NMS score threshold value. The runtime latency of the EfficientNMS plugin is sensitive to the score threshold used, so it's a good practice to set this value as high as possible, while still fulfilling your application requirements, to reduce latency as much as possible.
* `--legacy_plugins` allows falling back to older plugins on systems where a version lower than TensorRT 8.0 is installed. This will result in substantially slower inference times however.

### Build TensorRT Engine

#### FP16 Precision

To build the TensorRT engine file with FP16 precision, run:

```
python build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --precision fp16
```

The file `engine.trt` will be created, which can now be used to infer with TensorRT.

For best results, make sure no other processes are using the GPU during engine build, as it may affect the optimal tactic selection process.

#### INT8 Precision

To build and calibrate an engine for INT8 precision, run:

```
python build_engine.py \
    --onnx /path/to/model.onnx \
    --engine /path/to/engine.trt \
    --precision int8 \
    --calib_input /path/to/calibration/images \
    --calib_cache /path/to/calibration.cache
```

Where `--calib_input` points to a directory with several thousands of images. For example, this could be a subset of the training or validation datasets that were used for the model. It's important that this data represents the runtime data distribution relatively well, therefore, the more images that are used for calibration, the better accuracy that will be achieved in INT8 precision. For ImageNet networks, we have found that 25,000 images gives a good result.

The `--calib_cache` controls where the calibration cache file will be written to. This is useful to keep a cached copy of the calibration results. Next time you need to build the engine for the same network, if this file exists, it will skip the calibration step and use the cached values instead.

Run `python build_engine.py --help` for additional calibration options.

### Benchmark TensorRT Engine

Optionally, you can obtain execution timing information for the built engine by using the trtexec utility, as:

```
trtexec \
    --loadEngine=/path/to/engine.trt \
    --useCudaGraph --noDataTransfers \
    --iterations=100 --avgRuns=100
```

If it's not already in your `$PATH`, the trtexec binary is usually found in `/usr/src/tensorrt/bin/trtexec`, depending on your TensorRT installation method.

An inference benchmark will run, with GPU Compute latency times printed out to the console. Depending on the version of TensorRT, you should see something similar to:

```
GPU Compute Time: min = 3.43042 ms, max = 3.90247 ms, mean = 3.45948 ms, median = 3.44067 ms, percentile(99%) = 3.72531 ms
```

## Inference

To perform object detection on a set of images with TensorRT, run:

```
python infer.py \
    --engine /paht/to/engine.trt \
    --input /path/to/images \
    --output /path/to/output
```

Where the input path can be either a single image file, or a directory of jpg/png/bmp images.

The detection results will be written out to the specified output directory, consisting of a visualization image and a tab-separated results file for each input image processed.

For optimal performance, inference should be done in C++ instead of Python, to make use of CUDA Graphs to launch the inference request. A sample of this will be available soon.

The TensorRT engine built with this process can also be used with either [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server) or [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk).

## Validation

Given a validation dataset (such as COCO val2017 data) and ground truth annotations (such as instances_val2017.json), you can get the mAP metrics for the built TensorRT engine. This will use the mAP metrics calculation script from the AutoML EfficientDet repository on `https://github.com/google/automl`.

```
python eval_coco.py \
    --engine /path/to/engine.trt \
    --input /path/to/coco/val2017 \
    --annotations /path/to/coco/annotations/instances_val2017.json \
    --automl_path /path/to/automl
```

Where the `--automl_path` argument points to the root of the AutoML repository.

**NOTE:** mAP metrics are highly sensitive to NMS threshold. Using a high threshold will obviously reduce the mAP value. Ideally, this should run with a threshold of 0.00 or 0.01, but such a low threshold will impact the runtime performance of the EfficientNMS plugin. So you may need to build separate TensorRT engines for different purposes, one with a low threshold (like 0.01) dedicated for validation, and one with your application specific threshold (like 0.4) for deployment inference to minimimze latency. This is why we keep the NMS threshold as a configurable parameter in the TensorRT conversion script.
