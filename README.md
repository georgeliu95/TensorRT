# TensorRT

This repo contains the open source components of TensorRT. Included are the source code for NvInferPlugin (collection of TensorRT 
Plugins), Caffe and ONNX Parsers as well as a collection of sample applications demonstrating the capabilites of TensorRT. 

## Dependencies 

To build TensorRT Open Source Software (OSS), you will need the latest version of CUDA toolkit, cuBLAS and cuDNN installed, such as CUDA 10.1 + cudNN 7.5 (older versions may also work and are selected via `-DCUDA_VERSION=` and `-DCUDNN_VERSION=` respectively). Other dependencies include CUB, ONNX, and Protobuf (3.0.0 strict).

## Building

To build you must obtain the latest TensorRT 6.0.0 binary release from [NVidia developer zone](https://developer.nvidia.com/tensorrt). The required TensorRT libraries are `libnvinfer` and `libnvinfer_plugin`. 

### Quick Start Script

Download TensorRT-OSS and dependencies.
```
git clone https://github.com/nvidia/TensorRT
git submodule add https://github.com/onnx/onnx-tensorrt.git parsers/onnx
git submodule add https://github.com/protocolbuffers/protobuf.git third_party/protobuf
git submodule add https://github.com/NVlabs/cub.git third_party/cub
```

```
mkdir build
cd build 
cmake -DTRT_LIB_DIR=[PATH TO DIRECTORY CONTAINING TRT LIBS] ..
make
```

### Configurable Settings
#### Required

- `TRT_LIB_DIR`: Path to the TensorRT installation directory containing `libnvinfer` and `libnvinfer_plugin` libraries.

#### Optional 

- `TRT_BIN_DIR`: `[${CMAKE_BINARY_DIR}]` Location where binaries will be placed when compiled 

- `CMAKE_TOOLCHAIN_FILE`: Path to a toolchain file to conduct cross compilation

- `CMAKE_BUILD_TYPE`: `[Release] | Debug` Whether to build a debug build or a release build

- `CUDA_VERISON`: `[10.1]` Version of CUDA to use

- `CUDNN_VERSION`: `[7.5]` Version of cuDNN to use

- `PROTOBUF_VERSION`: `[3.0.0]` Version of Protobuf to use (note changing this will not configure CMake to use a system version of Protobuf, it will configure CMake to download that version and try to compile it)

- `BUILD_PARSERS`: `[ON] | OFF` Specify if the parsers should be built (if turned off, CMake will try to find precompiled versions of the parser libaries to use in compiling samples. First in `${TRT_LIB_DIR}` then on the system. If the build type is `Debug`  then it will prefer debug builds of the libraries before release versions if available)

- `BUILD_PLUGINS`: `[ON] | OFF` Specify if the plugins should be built (if turned off, CMake will try to find a precompiled version of the infer_plugin library to use in compiling samples. First in `${TRT_LIB_DIR}` then on the system. If the build type is `Debug` then it will prefer debug builds of the libraries before release versions if available)

- `BUILD_SAMPLES`  `[ON] | OFF` Specify if the samples should be built.


##### Options of limited applicability

- `NVINTERNAL`: `[OFF] | ON ` This is used by TensorRT engineering team for internal builds.
- `NVPARTNER`: `[OFF] | ON` This is used by NVidia partners with exclusive source access.
- `CUB_VERSION`: `[1.8.0]` Version of CUB to use.
- `PROTOBUF_INTERNAL_VERSION` `[10.0]` Version of protobuf to use, only applicable if `NVINTERNAL` is also enabled.
- `GPU_ARCHS`: ONNX parser GPU architectures to target.

## Known Issues

ONNX Parser insists on picking up a precompiled version of `libnvinfer_plugin`, since it looks for the library at configure time. 
This will be fixed as soon as the onnx-tensorrt's build system is updated. 