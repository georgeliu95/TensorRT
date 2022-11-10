# Introduction

This demo application ("demoDiffusion") showcases the acceleration of [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4) pipeline using TensorRT plugins.

# Setup

### Clone the TensorRT OSS repository

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/TensorRT/Public/oss.git -b dev/demodiffusion --single-branch
cd oss
```
> TODO - update following to GitHub repo for release

### Launch NVidia TensorRT container

Install nvidia-docker using [these intructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

```bash
docker run --rm -it --gpus all -v $PWD:/workspace nvcr.io/nvidia/tensorrt:22.10-py3 /bin/bash
```
NOTE: Alternatively, you can download and install TensorRT packages from [NVIDIA TensorRT Developer Zone](https://developer.nvidia.com/tensorrt).

### Build TensorRT plugins library

Follow [build instructions](https://github.com/NVIDIA/TensorRT/blob/main/README.md#building-tensorrt-oss) in the TensorRT OSS README document.

```bash
export TRT_OSSPATH=/workspace

cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_OUT_DIR=$PWD/out
cd plugin
make -j$(nproc)

export PLUGIN_LIBS="$TRT_OSSPATH/build/out/libnvinfer_plugin.so"
```

### Install packages required to run the Diffusion demo

```bash
cd  $TRT_OSSPATH/demo/Diffusion
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Create required output directories
mkdir -p onnx engine output
```

> NOTE: demoDiffusion has been tested on systems with NVIDIA A100, RTX3090, and RTX4090 GPUs, and the following software configuration.
```
cuda-python         11.8.1
diffusers           0.7.2
onnx                1.12.0
onnx-graphsurgeon   0.3.25
onnxruntime         1.13.1
polygraphy          0.43.1
tensorrt            8.5.1.7
tokenizers          0.13.2
torch               1.12.0+cu116
transformers        4.24.0
```

> NOTE: optionally install HuggingFace [accelerate](https://pypi.org/project/accelerate/) package for faster and less memory-intense model loading.


# Running demoDiffusion

### Review usage instructions under the help menu

```bash
python3 demo-diffusion.py --help
```

### Obtain HuggingFace access token

To download the model checkpoints for the Stable Diffusion pipeline, you will need a `read` access token. See instructions on how to generate it [here](https://huggingface.co/docs/hub/security-tokens).

```bash
export HF_TOKEN=<your access token>
```

### Generate an image guided by a single text prompt

```bash
# Download MHA and MHCA plugins and add them to the list of plugin libraries to be preloaded
wget http://tensorrt-rajeev/share/temp/sd/perflab/fmhaPlugin.so -O /tmp/fmhaPlugin.so
wget http://tensorrt-rajeev/share/temp/sd/perflab/fmhcaPlugin.so -O /tmp/fmhcaPlugin.so
export PLUGIN_LIBS="/tmp/fmhaPlugin.so:/tmp/fmhcaPlugin.so:$PLUGIN_LIBS"

LD_PRELOAD=${PLUGIN_LIBS} python3 demo-diffusion.py "a beautiful photograph of Mt. Fuji during cherry blossom" --hf-token=$HF_TOKEN -v
```
> TODO - remove extra step for MHA plugins

The above prompt `a beautiful photograph of Mt. Fuji during cherry blossom"` might generate and image similar to the following.


# Restrictions

### demoDiffusion
- Supports upto 8 simultaneous prompts (maximum batch size)
