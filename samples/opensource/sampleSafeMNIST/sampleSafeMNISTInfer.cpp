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
//! Command: ./sample_mnist_safe_infer [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

#include "safeInferCommon.h"

const std::string gSampleName = "TensorRT.sample_mnist_safe_infer";

using namespace safeInferCommon;

//!
//! \brief  The SampleSafeMNIST class implements the MNIST sample.
//!
//! \details It loads a prebuild safe engine and runs the TensorRT inference.
//!
class SampleSafeMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

public:
    SampleSafeMNIST(const samplesCommon::CaffeSampleParams& params)
        : mParams(params)
    {
    }
    
    //!
    //! \brief Runs the TensorRT inference engine for this sample.
    //!
    bool infer();

private:
    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer.
    //!
    bool processInput(const BufferManager& buffers, const std::string& inputTensorName, int inputFileIdx) const;

    //!
    //! \brief Verifies that the output is correct and prints it.
    //!
    bool verifyOutput(const BufferManager& buffers, const std::string& outputTensorName, int groundTruthDigit) const;

    samplesCommon::CaffeSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.
};

//!
//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer.
//!
bool SampleSafeMNIST::processInput(const BufferManager& buffers, const std::string& inputTensorName, int inputFileIdx) const
{
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];

    // Read the digit file according to the inputFileIdx.
    std::vector<uint8_t> fileData(inputH * inputW);
    readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print ASCII representation of digit.
    gLogInfo << "Input:\n";
    for (int i = 0; i < inputH * inputW; i++)
    {
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    gLogInfo << std::endl;

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorName));
    std::copy(fileData.begin(), fileData.end(), hostInputBuffer);

    return true;
}

//!
//! \brief Verifies that the output is correct and prints it.
//!
bool SampleSafeMNIST::verifyOutput(const BufferManager& buffers, const std::string& outputTensorName, int groundTruthDigit) const
{
    const float* prob = static_cast<const float*>(buffers.getHostBuffer(outputTensorName));

    // Print histogram of the output distribution.
    gLogInfo << "Output:\n";
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

        gLogInfo << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
    gLogInfo << std::endl;

    return (idx == groundTruthDigit && val > 0.9f);
}

//!
//! \brief Runs the TensorRT inference engine for this sample.
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleSafeMNIST::infer()
{
    // Load engine
    auto engine = loadEngine();
    if (engine.get() == nullptr)
    {
        gLogError << "Unable to load engine." << std::endl;
        return false;
    }

    // Get the input dimensions.
    mInputDims = engine->getBindingDimensions(0);

    // Create RAII buffer manager object.
    BufferManager buffers(engine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Pick a random digit to try to infer.
    std::random_device rd;
    std::default_random_engine generator{rd()};
    std::uniform_int_distribution<int> distribution(0, 9);
    const int digit = distribution(generator);

    // Read the input data into the managed buffers.
    // There should be just 1 input tensor.
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, mParams.inputTensorNames[0], digit))
    {
        return false;
    }
    
    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers.
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work.
    if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
    {
        return false;
    }
    // Asynchronously copy data from device output buffers to host output buffers.
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete.
    cudaStreamSynchronize(stream);

    // Release stream.
    cudaStreamDestroy(stream);

    // Check and print the output of the inference.
    // There should be just one output tensor.
    assert(mParams.outputTensorNames.size() == 1);
    bool outputCorrect = verifyOutput(buffers, mParams.outputTensorNames[0], digit);

    return outputCorrect;
}

//!
//! \brief Initializes members of the params struct using the command line args.
//!
samplesCommon::CaffeSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::CaffeSampleParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths.
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else //!< Use the data directory provided by the user.
    {
        params.dataDirs = args.dataDirs;
    }

    params.inputTensorNames.push_back("data");
    params.batchSize = 1;
    params.outputTensorNames.push_back("prob");

    return params;
}

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_mnist_safe_infer [-h or --help] [-d or --datadir=<path to data directory>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    samplesCommon::CaffeSampleParams params = initializeSampleParams(args);

    SampleSafeMNIST sample(params);
    gLogInfo << "Running a GPU inference engine for MNIST" << std::endl;

    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
