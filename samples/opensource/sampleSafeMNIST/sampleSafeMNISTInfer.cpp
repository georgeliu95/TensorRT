/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! \file sampleSafeMNISTInfer.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It uses the prebuilt TensorRT engine to run inference on an input image of a digit.
//! It can be run with the following command line:
//! Command: ./sample_mnist_safe_infer

#include "NvInferRTProxy.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

const std::string gSampleName = "TensorRT.sample_mnist_safe_infer";
const std::string INPUT_BLOB_NAME = "data";
const std::string OUTPUT_BLOB_NAME = "prob";

#define CHECK(status)                                                                                                  \
    {                                                                                                                  \
        if (status != 0)                                                                                               \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << status << std::endl;                                                      \
            abort();                                                                                                   \
        }                                                                                                              \
    }

//!
//! \brief Locate path to file by its filename. Will walk back MAX_DEPTH dirs from CWD to check for such a file path.
//!
inline std::string locateFile(const std::string& fileName)
{
    std::string file = "data/samples/mnist/" + fileName;
    const int MAX_DEPTH{10};
    bool found{false};

    for (int i = 0; i < MAX_DEPTH && !found; i++)
    {
        std::ifstream checkFile(file);
        found = checkFile.is_open();
        if (found)
        {
            break;
        }
        file = "../" + file; // Try again in parent dir.
    }

    if (!found)
    {
        std::cout << "Could not find " << fileName << " in data/samples/mnist/." << std::endl;
        std::cout << "&&&& FAILED" << std::endl;
        exit(EXIT_FAILURE);
    }
    return file;
}

//!
//! \brief Read data from a pgm file, then store to the buffer.
//!
inline void readPGMFile(const std::string& fileName, uint8_t* buffer, int inH, int inW)
{
    std::ifstream infile(fileName, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), inH * inW);
}

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
inline std::shared_ptr<T> infer_object(T* obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }

    return std::shared_ptr<T>(obj, InferDeleter());
}

//!
//! \brief Logger for TensorRT info/warning/errors.
//!
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != ILogger::Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

//!
//! \brief Helper function to get the element size.
//!
inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

//!
//! \brief Helper function to get the volume.
//!
inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

//!
//! \brief  Load a prebuilt TensorRT safe engine.
//!
std::shared_ptr<nvinfer1::ICudaEngine> loadEngine()
{
    const std::string& filename = "safe_mnist.engine";
    std::vector<char> gieModelStream;
    std::ifstream file(filename, std::ios::binary);
    if (!file.good())
    {
        std::cout << "[E] Could not open input engine file or file is empty. File name: " << filename << std::endl;
        return nullptr;
    }
    file.seekg(0, std::ifstream::end);
    auto size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    gieModelStream.resize(size);
    file.read(gieModelStream.data(), size);
    file.close();
    auto infer = infer_object(nvinfer1::createInferRuntime(gLogger));
    if (infer.get() == nullptr)
    {
        return nullptr;
    }
    auto engine = infer_object(infer->deserializeCudaEngine(gieModelStream.data(), size));

    return engine;
}

//!
//! \brief Reads the input data, preprocesses, and stores the result in a managed buffer.
//!
bool processInput(void* input, const std::string& inputTensorName, nvinfer1::Dims inputDims, const int inputFileIdx)
{
    const int inputH = inputDims.d[1];
    const int inputW = inputDims.d[2];

    // Read the digit file according to the inputFileIdx.
    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm"), fileData.data(), inputH, inputW);

    // Print ASCII representation of digit.
    std::cout << "[I] Input:\n";
    for (int i = 0; i < inputH * inputW; i++)
    {
        std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    std::cout << std::endl;

    float* hostInputBuffer = static_cast<float*>(input);
    std::copy(fileData.begin(), fileData.end(), hostInputBuffer);

    return true;
}

//!
//! \brief Verifies that the output is correct and prints it.
//!
bool verifyOutput(void* output, const std::string& outputTensorName, int groundTruthDigit)
{
    const float* prob = static_cast<const float*>(output);

    // Print histogram of the output distribution.
    std::cout << "[I] Output:\n";
    float val{0.0f};
    int idx{0};
    const int kDIGITS{10};

    for (int i = 0; i < kDIGITS; i++)
    {
        if (val < prob[i])
        {
            val = prob[i];
            idx = i;
        }

        std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    std::cout << std::endl;

    return (idx == groundTruthDigit && val > 0.9f);
}

//!
//! \brief Runs the TensorRT inference engine for this sample.
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool doInference(int batchSize)
{
    // Load engine
    auto engine = loadEngine();
    if (engine.get() == nullptr)
    {
        std::cout << "[E] Unable to load engine." << std::endl;
        return false;
    }

    // This sample only has one input and one output.
    assert(engine->getNbBindings() == 2);

    // Get the binding index according to the input/output tensor.
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME.c_str());
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME.c_str());

    // Get the binding dimensions according to the input/output index.
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

    // Get the input/output tensor volume.
    const size_t inputVol = volume(inputDims);
    const size_t outputVol = volume(outputDims);

    // Get the input/output element size.
    const size_t inElementSize = getElementSize(engine->getBindingDataType(inputIndex));
    const size_t outElementSize = getElementSize(engine->getBindingDataType(outputIndex));

    // Create GPU buffers
    void* buffers[2];
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * inputVol * inElementSize));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputVol * outElementSize));

    // Create host buffers
    void* hostBuffers[2];
    hostBuffers[inputIndex] = malloc(batchSize * inputVol * inElementSize);
    hostBuffers[outputIndex] = malloc(batchSize * outputVol * outElementSize);

    // Pick a random digit to try to infer.
    std::random_device rd;
    std::default_random_engine generator{rd()};
    std::uniform_int_distribution<int> distribution(0, 9);
    const int digit = distribution(generator);

    // Read the input data into the managed buffers.
    if (!processInput(hostBuffers[inputIndex], INPUT_BLOB_NAME, inputDims, digit))
    {
        return false;
    }

    // Create execution context.
    auto context = infer_object(engine->createExecutionContext());
    if (context.get() == nullptr)
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers.
    CHECK(cudaMemcpyAsync(buffers[inputIndex], hostBuffers[inputIndex], batchSize * inputVol * inElementSize, cudaMemcpyHostToDevice, stream));

    // Asynchronously enqueue the inference work.
    if (!context->enqueue(batchSize, buffers, stream, nullptr))
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers.
    CHECK(cudaMemcpyAsync(hostBuffers[outputIndex], buffers[outputIndex], batchSize * outputVol * outElementSize, cudaMemcpyDeviceToHost, stream));

    // Wait for the work in the stream to complete.
    cudaStreamSynchronize(stream);

    // Check and print the output of the inference.
    bool outputCorrect = verifyOutput(hostBuffers[outputIndex], OUTPUT_BLOB_NAME, digit);

    // Release stream and buffers.
    cudaStreamDestroy(stream);
    free(hostBuffers[inputIndex]);
    free(hostBuffers[outputIndex]);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    return outputCorrect;
}

int main(int argc, char** argv)
{
    auto cmdline = argv[0];
    std::cout << "&&&& RUNNING " << gSampleName << " # " << cmdline << std::endl;

    if (!doInference(1))
    {
        std::cout << "&&&& FAILED " << gSampleName << " # " << cmdline << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "&&&& PASSED " << gSampleName << " # " << cmdline << std::endl;
    return EXIT_SUCCESS;
}
