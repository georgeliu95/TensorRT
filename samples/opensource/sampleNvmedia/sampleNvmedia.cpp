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

//! \file SampleNvmedia.cpp
//! \brief This file contains the implementation of the nvmedia sample.
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "half.h"
#include "logger.h"

#include "NvInfer.h"

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
#include "nvmedia_dla.h"
#include "nvmedia_tensor.h"
#include "nvmedia_tensormetadata.h"
#include "nvmedia_core.h"
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

#include <random>
#include <utility>
#include <array>
#include <vector>
#include <string>

const std::string gSampleName = "TensorRT.sample_nvmedia";

bool isDLAHeader(const void* ptr)
{
    if (nullptr == ptr)
    {
        return false;
    }
    const char* p = static_cast<const char*>(ptr);
    return p[4] == 'N'
        && p[5] == 'V'
        && p[6] == 'D'
        && p[7] == 'A';
}

uint64_t volume(const nvinfer1::Dims& d)
{
    return static_cast<uint64_t>(std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>()));
}

enum class TensorType : int
{
    kINPUT = 0,
    kOUTPUT = 1,
};

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
class DlaContext
{
public:
    DlaContext() = default;

    ~DlaContext() { destroy(); }

    bool createDla()
    {
        dla = NvMediaDlaCreate();
        if (!dla)
        {
            gLogError << "DLA create failed" << std::endl;
            return false;
        }
        return true;
    }

    void destroyDla()
    {
        if (nullptr == dla)
        {
            return;
        }

        NvMediaStatus destroyStatus = NvMediaDlaDestroy(dla);

        if (destroyStatus != NVMEDIA_STATUS_OK)
        {
            gLogError << "DLA destroy failed" << std::endl;
        }
        dla = nullptr;
    }

    bool createDevice()
    {
        device = NvMediaDeviceCreate();
        if (!device)
        {
            gLogError << "Device create failed" << std::endl;
            return false;
        }
        return true;
    }

    void destroyDevice()
    {
        if (nullptr == device)
        {
            return;
        }

        NvMediaDeviceDestroy(device);
        device = nullptr;
    }

    bool createTensor(const nvinfer1::Dims& dims, TensorType type)
    {
        NVM_TENSOR_DEFINE_ATTR(tensorAttr);
        NVM_TENSOR_SET_ATTR_4D(tensorAttr, 1, dims.d[0], dims.d[1], dims.d[2], NCxHWx, FLOAT, 16, UNCACHED, NONE, 1);

        NvMediaTensor * tensor = NvMediaTensorCreate(device, tensorAttr, NVM_TENSOR_ATTR_MAX, 0);
        if (!tensor)
        {
            gLogError << "Input tensor create failed" << std::endl;
            return false;
        }
        if (type == TensorType::kINPUT)
        {
            tensorIn.push_back(tensor);
        }
        else
        {
            tensorOut.push_back(tensor);
        }
        return true;
    }

    void destroyTensor()
    {
        for (auto elem : tensorIn)
        {
            NvMediaTensorDestroy(elem);
        }

        tensorIn.clear();

        for (auto elem : tensorOut)
        {
            NvMediaTensorDestroy(elem);
        }

        tensorOut.clear();
    }

    template <typename T>
    bool fillInputTensor(const std::vector<T>& inputBuf, const int index)
    {
        NvMediaStatus status = NVMEDIA_STATUS_OK;
        NvMediaTensorSurfaceMap tensorMap = {};
        status = NvMediaTensorLock(tensorIn[index], NVMEDIA_TENSOR_ACCESS_WRITE, &tensorMap);

        if (status != NVMEDIA_STATUS_OK)
        {
            gLogError << "NvMediaTensorLock failed" << std::endl;
            return false;
        }

        unsigned int inputBufSize = inputBuf.size() * sizeof(T);

        if (tensorMap.size < inputBufSize)
        {
            gLogError << "Input tensor smaller than required size. tensorSize: " << tensorMap.size << ", requiredSize: " << inputBufSize << std::endl;
            return false;
        }

        std::copy(inputBuf.begin(), inputBuf.end(), reinterpret_cast<T*>(tensorMap.mapping));
        NvMediaTensorUnlock(tensorIn[index]);
        return true;
    }

    template <typename T>
    bool copyOutputTensor(std::vector<T>& outputBuf, const int index)
    {
        NvMediaStatus status = NVMEDIA_STATUS_OK;
        NvMediaTensorSurfaceMap tensorMap = {};
        status = NvMediaTensorLock(tensorOut[index], NVMEDIA_TENSOR_ACCESS_READ, &tensorMap);
        if (status != NVMEDIA_STATUS_OK)
        {
            gLogError << "NvMediaTensorLock failed" << std::endl;
            return false;
        }

        unsigned int expectedOutputSize = outputBuf.size() * sizeof(T);

        std::copy(reinterpret_cast<int8_t*>(tensorMap.mapping), reinterpret_cast<int8_t*>(tensorMap.mapping) + expectedOutputSize, outputBuf.data());
        NvMediaTensorUnlock(tensorOut[index]);
        return true;
    }

    bool submit()
    {
        NvMediaStatus status = NVMEDIA_STATUS_OK;

        constexpr int maxDlaDataSize = 10;
        std::array<NvMediaDlaData, maxDlaDataSize> inputData;
        std::array<NvMediaDlaData, maxDlaDataSize> outputData;

        NvMediaDlaArgs inputArgs;
        NvMediaDlaArgs outputArgs;

        for (unsigned int i = 0; i < tensorIn.size(); i++)
        {
            inputData[i].type = NVMEDIA_DLA_DATA_TYPE_TENSOR;
            inputData[i].pointer.tensor = tensorIn[i];
        }
        inputArgs.numArgs = tensorIn.size();
        inputArgs.dlaData = inputData.data();

        for (unsigned int i = 0; i < tensorOut.size(); i++)
        {
            outputData[i].type = NVMEDIA_DLA_DATA_TYPE_TENSOR;
            outputData[i].pointer.tensor = tensorOut[i];
        }
        outputArgs.numArgs = tensorOut.size();
        outputArgs.dlaData = outputData.data();

        status = NvMediaDlaSubmitTimeout(dla, &inputArgs, &outputArgs, 30000);
        if (status != NVMEDIA_STATUS_OK)
        {
            gLogError << "NvMediaDlaSubmitTimeout failed. status:" << status << std::endl;
            return false;
        }

        gLogInfo << "DLA submit successfully" << std::endl;

        NvMediaTensorTaskStatus taskStatus = {};
        status = NvMediaTensorGetStatus(tensorOut[0], NVMEDIA_TENSOR_TIMEOUT_INFINITE, &taskStatus);
        if (status != NVMEDIA_STATUS_OK)
        {
            gLogError << "NvMediaTensorGetStatus failed. status:" << status << std::endl;
            return false;
        }
        gLogInfo << "NvMediaTensorGetStatus successful" << std::endl;
        gLogInfo << "Operation duration: " << taskStatus.durationUs << " us" << std::endl;
        if (taskStatus.status != NVMEDIA_STATUS_OK)
        {
            status = taskStatus.status;
            gLogError << "Engine returned error." << std::endl;
            return false;
        }

        return true;
    }

    void destroy()
    {
        destroyDla();
        destroyTensor();
        destroyDevice();
    }

    NvMediaDla * getDla()
    {
        return dla;
    }

private:
    NvMediaDla * dla{nullptr};
    NvMediaDevice *device{nullptr};
    std::vector<NvMediaTensor*> tensorIn;
    std::vector<NvMediaTensor*> tensorOut;
};
#endif

//!
//! \brief  The SampleNvmedia class implements the nvmeidia sample.
//!
//! \details It creates the network using a single conv layer.
//!
class SampleNvmedia
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleNvmedia(int batchSize) : mBatchSize(batchSize) { assert(mBatchSize >= 1); }

    //!
    //! \brief Builds the network engine.
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample.
    //!
    bool infer() const;

    //!
    //! \brief Used to clean up any state created in the sample class.
    //!
    bool teardown() { return true; }

    //!
    //! \brief Randomly intitializes buffer.
    //!
    template <typename T>
    void randomInit(std::vector<T>& buffer) const;

    //!
    //! \brief Verifies that the output.
    //!
    template <typename T>
    bool verifyOutput(const vector<T>& inputBufA, const vector<T>& inputBufB, const vector<T>& outputBuf) const;

private:
    //!
    //! \brief Create a single layer Network and marks the output layers.
    //!
    void constructNetwork(SampleUniquePtr<nvinfer1::INetworkDefinition>& network) const;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network.

public:

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    int mBatchSize{1};
};

//!
//! \brief Creates the network, configures the builder, and creates the network engine.
//!
//! \details This function creates a network and builds an engine to run in DLA safe mode.
//! The network consists of only one elementwise sum layer with FP16 precision.
//! 
//! \return Returns true if the engine was created successfully and false otherwise.
//!
bool SampleNvmedia::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    constructNetwork(network);

    builder->setMaxBatchSize(mBatchSize);

    mEngine.reset();

    config->setMaxWorkspaceSize(256_MiB);
    config->setFlag(BuilderFlag::kFP16);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);

    samplesCommon::enableDLA(builder.get(), config.get(), 0);

    config->clearFlag(BuilderFlag::kGPU_FALLBACK);
    config->setEngineCapability(EngineCapability::kSAFE_DLA);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    if (!mEngine)
    {
        return false;
    }

    mInputDims = network->getInput(0)->getDimensions();

    mOutputDims = network->getOutput(0)->getDimensions();

    return true;
}

//!
//! \brief Create the single layer Network and marks the output layers.
//!
void SampleNvmedia::constructNetwork(SampleUniquePtr<nvinfer1::INetworkDefinition>& network) const
{
    nvinfer1::Dims inputDims{3, {32, 32, 32}, {}};

    auto inA = network->addInput("inputA", DataType::kHALF, inputDims);
    auto inB = network->addInput("inputB", DataType::kHALF, inputDims);

    auto layer = network->addElementWise(*inA, *inB, ElementWiseOperation::kSUM);
    ITensor* out = layer->getOutput(0);

    out->setName("output");
    network->markOutput(*out);
    out->setType(DataType::kHALF);
}

//!
//! \brief Runs the TensorRT inference engine for this sample.
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleNvmedia::infer() const
{
#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    auto mem = SampleUniquePtr<nvinfer1::IHostMemory>(mEngine->serialize());
    if (mem.get() == nullptr)
    {
        gLogError << "Engine serialization failed" << std::endl;
        return false;
    }

    auto context = SampleUniquePtr<DlaContext>(new DlaContext());

    if (!isDLAHeader(reinterpret_cast<char*>(mem->data())))
    {
        gLogError << "Invalid DLA header" << std::endl;
        return false;
    }

    status = NvMediaDlaLoadFromMemory(context->getDla(), reinterpret_cast<uint8_t*>(mem->data()), mem->size(), 0, 1);
    if (status != NVMEDIA_STATUS_OK)
    {
        gLogError << "Failure to load program." << std::endl;
        return false;
    }

    int inputBufSize = volume(mInputDims);
    int expectedOutputSize = volume(mOutputDims);

    std::vector<half_float::half> inputBufA(inputBufSize);
    std::vector<half_float::half> inputBufB(inputBufSize);
    std::vector<half_float::half> outputBuf(expectedOutputSize);

    if (!context->createDla())
    {
        return false;
    }

    if (!context->createDevice())
    {
        return false;
    }

    if (!context->createTensor(mInputDims, TensorType::kINPUT))
    {
        return false;
    }

    if (!context->createTensor(mInputDims, TensorType::kINPUT))
    {
        return false;
    }

    if (!context->createTensor(mInputDims, TensorType::kOUTPUT))
    {
        return false;
    }

    randomInit(inputBufA);
    randomInit(inputBufB);

    if (!context->fillInputTensor(inputBufA, 0))
    {
        return false;
    }

    if (!context->fillInputTensor(inputBufB, 1))
    {
        return false;
    }

    if (!context->submit())
    {
        return false;
    }

    if (!context->copyOutputTensor(outputBuf, 0))
    {
        return false;
    }
    
    if (!verifyOutput(inputBufA, inputBufB, outputBuf))
    {
        return false;
    }
#endif
    return true;
}

//!
//! \brief Randomly initializes buffer.
//!
template <typename T>
void SampleNvmedia::randomInit(std::vector<T>& buffer) const
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, 63);

    auto gen = [&dist, &mt] () { return T(dist(mt)); };

    std::generate(buffer.begin(), buffer.end(), gen);
}

//!
//! \brief Verifies that the output is correct and prints it.
//!
template<typename T>
bool SampleNvmedia::verifyOutput(const vector<T>& inputBufA, const vector<T>& inputBufB, const vector<T>& outputBuf) const
{
    assert(outputBuf.size() == volume(mInputDims));

    std::vector<T> reference(outputBuf.size());

    std::transform(inputBufA.begin(), inputBufA.end(), inputBufB.begin(), reference.begin(), std::plus<T>());
    return reference == outputBuf;
}

//!
//! \brief Prints the help information for running this sample.
//!
void printHelpInfo()
{
    gLogInfo << "Usage: ./sample_nvmedia [-h or --help]\n";
    gLogInfo << "--help          Display help information\n";
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments " << std::endl;
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

#if (!defined(__ANDROID__) && defined(__aarch64__)) || defined(__QNX__)

    SampleNvmedia sample(1);

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }

    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }

    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }
#else
    gLogError << "Unsupported platform, please make sure it is running on aarch64, QNX or android" << std::endl;
#endif
    return gLogger.reportPass(sampleTest);
}
