cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(infer_c LANGUAGES CXX)
find_package(CUDA)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.2.3
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

include($ENV{TRT_SOURCE}/cmake/modules/set_ifndef.cmake)
set_ifndef(TRT_INC_DIR $ENV{TRT_SOURCE}/include)
set_ifndef(TRT_LIB_DIR $ENV{TRT_SOURCE}/build/out)

include_directories(
    infer_c
    ${CUDA_INCLUDE_DIRS}
    ${TRT_INC_DIR}
)

link_directories(
    ${TRT_LIB_DIR}
)

pybind11_add_module(infer_c 
    infer_c/infer_c.cpp
    infer_c/logging.cpp
)
target_link_libraries(infer_c PRIVATE
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
)

add_executable(perf
    infer_c/perf.cpp
    infer_c/logging.cpp
)

target_link_libraries(perf 
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
)

