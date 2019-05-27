# TensorRT

This repo contains the open source components of TensorRT. Included is the source code for NvInferPlugin (a collection of TensorRT 
Plugins), the source code for the Caffe and ONNX Parsers as well as a collection of sample applications demonstrating the capabilites
of TensorRT. 

## Dependencies 

To build TensorRT, you need CUDA 10.1 (you can use older versions by setting `-DCUDA_VERSION` equal to the desired version), 
cuDNN 7.5 (over versions can be used by setting `-DCUDNN_VERSION`) and cuBLAS. Other dependencies include CUB, ONNX, and Protobuf (3.0.0 strict) 
though these dependencies are already handled via either git submodules or CMake

## Building

To build you must have a copy of TensorRT 6.0.0. The only required libraries from TensorRT is `libnvinfer` and `libnvparsers` (unless code was
obtained under an NDA). 

### Quick Start Script

```
git clone [URL OF REPO]
git submodules update --init --recursive
mkdir build
cd build 
cmake -DTRT_LIB_DIR=[PATH TO DIRECTORY CONTAINING TRT LIBS] ..
make
```

### Configurable Settings
#### Required

- `TRT_LIB_DIR`: Path to the directory container `libnvinfer` and `libnvparsers`

#### Optional 

- `TRT_BIN_DIR`: `[${CMAKE_BINARY_DIR}]` Location where binaries will be placed when compiled 

- `CMAKE_TOOLCHAIN_FILE`: Path to a toolchain file to conduct cross compilation

- `CMAKE_BUILD_TYPE`: `[Release] | Debug` Whether to build a debug build or a release build

- `CUDA_VERISON`: `[10.1]` Version of CUDA to use

- `CUDNN_VERSION`: `[7.5]` Version of cuDNN to use

- `PROTOBUF_VERSION`: `[3.0.0]` Version of Protobuf to use (note changing this will not configure CMake to use a system version of Protobuf, 
                                it will configure CMake to download that version and try to compile it)

- `BUILD_PARSERS`: `[ON] | OFF` Should the parsers be built (if turned off, CMake will try to find precompiled versions of the parser
                                libaries to use in compiling samples. First in `${TRT_LIB_DIR}` then on the system. If the build type is `Debug` 
                                then it will prefer debug builds of the libraries before release versions if available)

- `BUILD_PLUGINS`: `[ON] | OFF` Should the plugins be built (if turned off, CMake will try to find a precompiled version of the infer_plugin 
                                libary to use in compiling samples. First in `${TRT_LIB_DIR}` then on the system. If the build type is `Debug` 
                                then it will prefer debug builds of the libraries before release versions if available)

- `BUILD_SAMPLES`  `[ON] | OFF` Should the samples be built


##### Options of limited applicability

- `NVINTERNAL`: `[OFF] | ON ` This is to be used by NVIDIANs to build TensorRT, it likely will not work outside of NVIDIA
- `NDA_RELEASE`: `[OFF] | ON` This is to be used if you recieved this code under an NDA and as such have the source code for the UFF parser  
- `CUB_VERSION`: `[1.8.0]` Version of CUB to use
- `PROTOBUF_INTERNAL_VERSION` `[10.0]` Version of protobuf to use, only is used if `NVINTERNAL` is on (hence only useful to NVIDIANs)
- `GPU_ARCHS`: ONNX parser GPU archs

## Known Issues

ONNX Parser insists on picking up a precompiled version of `libnvinfer_plugin`, since it looks for the library at configure time. 
This will be fixed as soon as the onnx-tensorrt's build system is updated. 