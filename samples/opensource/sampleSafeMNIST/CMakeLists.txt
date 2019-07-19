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
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(SafeMNIST LANGUAGES CXX C)

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if(NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)
# Build targets
set_ifndef(BUILD_TARGET_NAME sample_mnist_safe_build)
set_ifndef(INFER_TARGET_NAME sample_mnist_safe_infer)
# Default settings
set_ifndef(CUDA_VERSION 10.0)
set_ifndef(CUDA_ROOT /usr/local/cuda-${CUDA_VERSION})
set_ifndef(CUDNN_ROOT ${CUDA_ROOT})
set_ifndef(TRT_LIB_DIR /usr/lib/x86_64-linux-gnu)
set_ifndef(TRT_INC_DIR /usr/include/x86_64-linux-gnu)

# Find dependencies.
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

find_path(_CUDA_INC_DIR cuda_runtime_api.h HINTS ${CUDA_ROOT} PATH_SUFFIXES include)
set_ifndef(CUDA_INC_DIR ${_CUDA_INC_DIR})

find_library(_CUDART_LIB cudart HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
set_ifndef(CUDART_LIB ${_CUDART_LIB})

find_library(_TRT_INC_DIR NvInfer.h HINTS ${TRT_INC_DIR} PATH_SUFFIXES include x86_64-linux-gnu)
set_ifndef(TRT_INC_DIR ${_TRT_INC_DIR})

find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

find_library(_NVINFERSAFE_LIB nvinfer_safe HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVINFERSAFE_LIB ${_NVINFERSAFE_LIB})

find_library(_NVPARSERS_LIB nvparsers HINTS ${TRT_LIB_DIR} PATH_SUFFIXES lib lib64 x86_64-linux-gnu)
set_ifndef(NVPARSERS_LIB ${_NVPARSERS_LIB})

get_filename_component(SAMPLES_DIR ../ ABSOLUTE)

# -------- BUILDING THE BUILD SAMPLE --------

add_executable(${BUILD_TARGET_NAME}
    sampleSafeMNISTBuild.cpp
    ${SAMPLES_DIR}/common/logger.cpp
    )

# Add include directories
target_include_directories(${BUILD_TARGET_NAME}
    PUBLIC ${TRT_INC_DIR}
    PUBLIC ${CUDA_INC_DIR}
    PRIVATE ${SAMPLES_DIR}/common
    )

target_link_libraries(${BUILD_TARGET_NAME}
    ${CUDART_LIB}
    ${NVINFER_LIB}
    ${NVPARSERS_LIB}
    )

# -------- BUILDING THE INFER SAMPLE --------

add_executable(${INFER_TARGET_NAME}
    sampleSafeMNISTInfer.cpp
    )

# Add include directories
target_include_directories(${INFER_TARGET_NAME}
    PUBLIC ${TRT_INC_DIR}
    PUBLIC ${CUDA_INC_DIR}
    )

option(TRT_SAFE_BUILD "Build with ISO26262-compliant TensorRT library" OFF)
# Force safe mode in release build
if (CMAKE_BUILD_TYPE MATCHES RELEASE)
    message("Setting TRT_SAFE_BUILD=ON in release build")
    set(TRT_SAFE_BUILD ON)
endif()

if (TRT_SAFE_BUILD)
    target_compile_definitions(${INFER_TARGET_NAME} PRIVATE TRT_SAFE=1)
    target_link_libraries(${INFER_TARGET_NAME}
        ${CUDART_LIB}
        ${NVINFERSAFE_LIB}
        )
else()
    target_compile_definitions(${INFER_TARGET_NAME} PRIVATE TRT_SAFE=0)
    target_link_libraries(${INFER_TARGET_NAME}
        ${CUDART_LIB}
        ${NVINFER_LIB}
        )
endif()
