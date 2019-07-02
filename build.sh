#!/bin/bash

#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ARCH="x86_64-linux"
CUDA=10.1
CUDNN=7.5
TRT_LOCAL_SM=61
BUILD_TYPE="release"
TRT_SOURCE=$PWD
TOOLCHAIN=

ARGS="$@"

usage()
{
    echo "build.sh usage"
    echo "Arguments: "
    echo -e "\t TRT_SOURCE      Path to OSS source code"
    echo -e "\nOptional Args:"
    echo "CUDA             cuda version to build against CUDA=10.1"
    echo "CUDNN            cuDNN version to build against e.g. CUDNN=7.5"
    echo "ARCH             Architecture for compilation  e.g. x86_64-linux | aarch64-linux | aarch64-qnx"
    echo "BUILD_TYPE       Build type for compilation e.g. release | debug (defaults to release)"
    echo "TRT_LOCAL_SM     Local SM, Volta supports faster builds"
    exit -1
}

for args in $ARGS
do
    set $(echo ${args} | awk -F '=' '{print $1" "$2}')
    arg=$1
    val=$2
    case $arg in
        TRT_SOURCE)
                TRT_SOURCE=$val
                ;;
        CUDA)
                CUDA=$val
                ;;
        CUDNN)
                CUDNN=$val
                ;;
        ARCH)
                ARCH=$val
                ;;
        BUILD_TYPE)
                BUILD_TYPE=$val
                ;;
        TRT_LOCAL_SM)
                TRT_LOCAL_SM=$val
                ;;
        --help)
                usage
                ;;
        *)
               echo "Command line argument $arg not supported"
               usage
               ;;
    esac
done

if [ "${BUILD_TYPE}" == "debug" ]; then
    CMAKE_BUILD_TYPE="Debug"
else
    CMAKE_BUILD_TYPE="Release"
fi

MAKE_BUILD_ARG=
CMAKE_ARGS=

if [[ ${ARCH} = "aarch64-linux" ]]; then
    MAKE_BUILD_ARG="${BUILD_TYPE}_aarch64"
    TOOLCHAIN=${PWD}/cmake/toolchains/cmake_aarch64.toolchain
    CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN}"
elif [[ ${PLATFORM} = "aarch64-qnx" ]]; then
    MAKE_BUILD_ARG="${BUILD_TYPE}_qnx"
    TOOLCHAIN=${PWD}/cmake/toolchains/cmake_qnx.toolchain
    CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN}"
else
    MAKE_BUILD_ARG="${BUILD_TYPE}_x86_64"
fi

# Compile the P4 trunk
COMPILE_ARGS=
if [[ ! -z ${CUDA} ]]; then
    COMPILE_ARGS="${COMPILE_ARGS} CUDA=cuda-${CUDA}"
fi

if [[ ! -z ${CUDNN} ]]; then
    COMPILE_ARGS="${COMPILE_ARGS} CUDNN=${CUDNN}"
fi

echo "Compiling P4 trunk"
echo "make -j$(nproc) ${MAKE_BUILD_ARG} ${COMPILE_ARGS} TRT_LOCAL_SM=${TRT_LOCAL_SM}"
make -j$(nproc) ${MAKE_BUILD_ARG} ${COMPILE_ARGS} TRT_LOCAL_SM=${TRT_LOCAL_SM}

echo "Compiling OSS components"
if [[ -d ${TRT_SOURCE}/build/cmake ]]; then
    rm -rf ${TRT_SOURCE}/build/cmake/*
fi

echo "mkdir -p ${TRT_SOURCE}/build/cmake && cd ${TRT_SOURCE}/build/cmake"
mkdir -p ${TRT_SOURCE}/build/cmake && cd ${TRT_SOURCE}/build/cmake

# Workaround for CUDA-9.x - build PPS independently
if [ "$CUDA" = "9.0" ] || [ "$CUDA" = "9.1" ] || [ "$CUDA" = "9.2" ] ; then
  echo "# Build Plugin Library"
  echo "cmake -DNVINTERNAL=OFF -DTRT_LIB_DIR=${TENSORRT_ROOT}/lib -DTRT_BIN_DIR=`pwd`/out -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCUDA_VERSION=${CUDA} -DCUDNN_VERSION=${CUDNN} -DBUILD_PLUGINS=ON -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=OFF ../.."
  cmake \
    -DNVINTERNAL=OFF \
    -DBUILD_PLUGINS=ON \
    -DBUILD_PARSERS=OFF \
    -DBUILD_SAMPLES=OFF \
    -DTRT_LIB_DIR=${TENSORRT_ROOT}/lib \
    -DTRT_BIN_DIR=`pwd`/out \
    -DCUDNN_ROOT_DIR=${CUDNN_ROOT} \
    -DCUB_ROOT_DIR=${CUB_ROOT} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCUDA_VERSION=${CUDA} \
    -DCUDNN_VERSION=${CUDNN} \
    ${CMAKE_ARGS} \
    ../..
  echo "make -j$(nproc) all"
  make -j$(nproc) all
  echo "# Build Parser Libraries"
  echo "cmake -DNVINTERNAL=OFF -DTRT_LIB_DIR=${TENSORRT_ROOT}/lib -DTRT_BIN_DIR=`pwd`/out -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCUDA_VERSION=${CUDA} -DCUDNN_VERSION=${CUDNN} -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=ON -DBUILD_SAMPLES=OFF ../.."
  cmake \
    -DNVINTERNAL=OFF \
    -DBUILD_PLUGINS=OFF \
    -DBUILD_PARSERS=ON \
    -DBUILD_SAMPLES=OFF \
    -DTRT_LIB_DIR=${TENSORRT_ROOT}/lib \
    -DTRT_BIN_DIR=`pwd`/out \
    -DCUDNN_ROOT_DIR=${CUDNN_ROOT} \
    -DCUB_ROOT_DIR=${CUB_ROOT} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCUDA_VERSION=${CUDA} \
    -DCUDNN_VERSION=${CUDNN} \
    ${CMAKE_ARGS} \
    ../..
  echo "make -j$(nproc) all"
  make -j$(nproc) all
  echo "# Build Samples"
  echo "cmake -DNVINTERNAL=OFF -DTRT_LIB_DIR=${TENSORRT_ROOT}/lib -DTRT_BIN_DIR=`pwd`/out -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCUDA_VERSION=${CUDA} -DCUDNN_VERSION=${CUDNN} -DBUILD_PLUGINS=OFF -DBUILD_PARSERS=OFF -DBUILD_SAMPLES=ON ../.."
  cmake \
    -DNVINTERNAL=OFF \
    -DBUILD_PLUGINS=OFF \
    -DBUILD_PARSERS=OFF \
    -DBUILD_SAMPLES=ON \
    -DTRT_LIB_DIR=${TENSORRT_ROOT}/lib \
    -DTRT_BIN_DIR=`pwd`/out \
    -DCUDNN_ROOT_DIR=${CUDNN_ROOT} \
    -DCUB_ROOT_DIR=${CUB_ROOT} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCUDA_VERSION=${CUDA} \
    -DCUDNN_VERSION=${CUDNN} \
    ${CMAKE_ARGS} \
    ../..
  echo "make -j$(nproc) all"
  make -j$(nproc) all
else
  echo "# Build all OSS components"
  echo "cmake -DNVINTERNAL=OFF -DTRT_LIB_DIR=${TENSORRT_ROOT}/lib -DTRT_BIN_DIR=`pwd`/out -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCUDA_VERSION=${CUDA} -DCUDNN_VERSION=${CUDNN} ../.."
  cmake \
    -DNVINTERNAL=OFF \
    -DBUILD_PLUGINS=ON \
    -DBUILD_PARSERS=ON \
    -DBUILD_SAMPLES=ON \
    -DTRT_LIB_DIR=${TENSORRT_ROOT}/lib \
    -DTRT_BIN_DIR=`pwd`/out \
    -DCUDNN_ROOT_DIR=${CUDNN_ROOT} \
    -DCUB_ROOT_DIR=${CUB_ROOT} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCUDA_VERSION=${CUDA} \
    -DCUDNN_VERSION=${CUDNN} \
    ${CMAKE_ARGS} \
    ../..
  echo "make -j$(nproc) all"
  make -j$(nproc) all
fi

