#!/bin/bash

ARCH="x86_64-linux"
CUDA=10.1
CUDNN=7.5
TRT_LOCAL_SM=61
BUILD_TYPE="release"
CMAKE_BUILD_TYPE="Release"
OSS_PATH=$PWD
BUILD_ROOT=$PWD
TOOLCHAIN=

ARGS="$@"

usage()
{
    echo "compile.sh usage"
    echo "Arguments: "
    echo -e "\t OSS_PATH      Path to OSS source code"
    echo -e "\nOptional Args:"
    echo "CUDA             Cuda build for compilation e.g. CUDA=10.1"
    echo "CUDNN            CUDNN build for compilation e.g. CUDNN=7.5"
    echo "ARCH             Architecture for compilation  e.g. x86_64-linux | aarch64-linux | aarch64-qnx (defaults to x86_64-linux)"
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
        OSS_PATH)
                OSS_PATH=$val
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
if [[ -d ${OSS_PATH}/build/cmake ]]; then
    rm -rf ${OSS_PATH}/build/cmake/*
fi

echo "mkdir -p ${OSS_PATH}/build/cmake && cd ${OSS_PATH}/build/cmake"
mkdir -p ${OSS_PATH}/build/cmake && cd ${OSS_PATH}/build/cmake

echo "cmake -DNVINTERNAL=ON -DTRT_LIB_DIR=${BUILD_ROOT}/build/${ARCH} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCUDA_VERSION=${CUDA} -DCUDNN_VERSION=${CUDNN} ../.."
cmake \
    -DNVINTERNAL=ON \
    -DTRT_LIB_DIR=${BUILD_ROOT}/build/${ARCH} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCUDA_VERSION=${CUDA} \
    -DCUDNN_VERSION=${CUDNN} \
    ${CMAKE_ARGS} \
    ../..

echo "make -j$(nproc) all"
make -j$(nproc) all
