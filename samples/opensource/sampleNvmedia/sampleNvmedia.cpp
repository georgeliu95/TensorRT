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

//! \file SampleNvMedia.cpp
//! \brief This file contains the implementation of the NvMedia sample.
//!

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "half.h"
#include "logger.h"

#include "NvInfer.h"
#include "dla/dlaFlags.h"

// ENABLE_DLA_API is defined in common.h
#if ENABLE_DLA_API

#include "nvmedia_dla.h"
#include "nvmedia_tensor.h"
#include "nvmedia_tensormetadata.h"

// Guard code that uses API with version >= 3.0
#if NVMEDIA_DLA_VERSION_MAJOR >= 3 && !ENABLE_NVMEDIA_L4T_WAR
#define ENABLE_DLA_API_API_3_0 1
#endif // NVMEDIA_DLA_VERSION_MAJOR >= 3 && !ENABLE_NVMEDIA_L4T_WAR

#endif // ENABLE_DLA_API

#if ENABLE_DLA_API_API_3_0

#include "nvmedia_tensor_nvscibuf.h"
#include "nvscibuf.h"

#endif // ENABLE_DLA_API_API_3_0

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <array>
#include <random>
#include <string>
#include <utility>
#include <vector>

const std::string gSampleName = "TensorRT.sample_nvmedia";

bool isDLAHeader(const void* ptr)
{
    CHECK_RETURN(ptr, false);
    const char* p = static_cast<const char*>(ptr);
    return p[4] == 'N' && p[5] == 'V' && p[6] == 'D' && p[7] == 'A';
}

uint64_t volume(const nvinfer1::Dims& d)
{
    return static_cast<uint64_t>(std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>()));
}

#if ENABLE_DLA_API_API_3_0

//!
//! \brief A set of APIs to manage NvSciBufAttrList
//!
class NvSciBufAttrListManager
{
public:
    NvSciBufAttrListManager() = default;

    bool createAttrListTensor(NvSciBufModule module)
    {
        return NvSciError_Success == NvSciBufAttrListCreate(module, &mUnreconciledAttrListTensor);
    }

    bool setAttrListTensor(NvSciBufAttrKeyValuePair* pairArray, size_t nbPair)
    {
        return NvSciError_Success == NvSciBufAttrListSetAttrs(mUnreconciledAttrListTensor, pairArray, nbPair);
    }

    const NvSciBufAttrList& getAttrListTensor() const
    {
        return mUnreconciledAttrListTensor;
    }

    bool reconcile()
    {
        NvSciBufAttrList unreconciledAttrList[2] = {mUnreconciledAttrListTensor, nullptr};
        freeNvSciBufAttrList(mReconciledAttrList);
        freeNvSciBufAttrList(mConflictAttrList);
        return (NvSciError_Success == NvSciBufAttrListReconcile(
                    unreconciledAttrList, 1U, &mReconciledAttrList, &mConflictAttrList));
    }

    bool objAlloc(NvSciBufObj* sciBufObj)
    {
        return NvSciError_Success == NvSciBufObjAlloc(mReconciledAttrList, sciBufObj);
    }

    void destroy() noexcept
    {
        freeNvSciBufAttrList(mReconciledAttrList);
        freeNvSciBufAttrList(mUnreconciledAttrListTensor);
        freeNvSciBufAttrList(mConflictAttrList);
    }

protected:
    void freeNvSciBufAttrList(NvSciBufAttrList& nvSciBufAttrList)
    {
        if (nvSciBufAttrList)
        {
            NvSciBufAttrListFree(nvSciBufAttrList);
            nvSciBufAttrList = nullptr;
        }
    }

private:
    NvSciBufAttrList mReconciledAttrList{nullptr};
    NvSciBufAttrList mUnreconciledAttrListTensor{nullptr};
    NvSciBufAttrList mConflictAttrList{nullptr};
};

//!
//! \brief A set of APIs to use NvSciBuf for buffer allocation.
//!
class NvSciBuf
{
public:
    bool initialize()
    {
        CHECK_RETURN(NvSciError_Success == NvSciBufModuleOpen(&mNvSciBufModule), false);
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaTensorNvSciBufInit(), false);
        return true;
    }

    bool allocate(NvMediaDevice* device, const NvMediaDlaTensorDescriptor& inDesc, NvMediaTensor*& nvmTensor)
    {
        nvmTensor = nullptr;
        std::vector<NvMediaTensorAttr> tensorAttrs(NVM_TENSOR_ATTR_MAX);

        std::for_each(tensorAttrs.begin(), tensorAttrs.end(), [&](NvMediaTensorAttr& attr) 
        {
            attr.type = static_cast<NvMediaTensorAttrType>(&attr - &tensorAttrs[0]);
            attr.value = 0;
        });

        for (unsigned i = 0; i < inDesc.numAttrs; ++i)
        {
            tensorAttrs[inDesc.tensorAttrs[i].type] = inDesc.tensorAttrs[i];
        }

        NvSciBufObj tensorSciBufObj{nullptr};
        CHECK_RETURN(allocateNvSciBuf(
            mNvSciBufModule, device, &tensorSciBufObj, tensorAttrs.data(), tensorAttrs.size(), &nvmTensor), false);

        mTensorSciBufObj.emplace_back(tensorSciBufObj);

        return (nullptr != nvmTensor);
    }

    void destroy() noexcept
    {
        for_each(mTensorSciBufObj.begin(), mTensorSciBufObj.end(), [](NvSciBufObj& iter) { NvSciBufObjFree(iter); });
        mTensorSciBufObj.clear();

        NvMediaTensorNvSciBufDeinit();

        NvSciBufModuleClose(mNvSciBufModule);
        mNvSciBufModule = nullptr;
    }

protected:
    static bool allocateNvSciBuf(NvSciBufModule nvSciBufModule, NvMediaDevice* device, NvSciBufObj* tensorSciBufObj,
        NvMediaTensorAttr* tensorAttr, uint32_t numTensorAttr, NvMediaTensor** nvmTensor)
    {
        auto attrListManager = makeObjGuard<NvSciBufAttrListManager>(new NvSciBufAttrListManager());
        NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
        NvSciBufAttrKeyValuePair attrKvp = {NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm)};
        CHECK_RETURN(attrListManager->createAttrListTensor(nvSciBufModule), false);
        CHECK_RETURN(attrListManager->setAttrListTensor(&attrKvp, 1), false);

        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaTensorFillNvSciBufAttrs(
            device, tensorAttr, numTensorAttr, 0, attrListManager->getAttrListTensor()), false);

        CHECK_RETURN(attrListManager->reconcile(), false);
        CHECK_RETURN(attrListManager->objAlloc(tensorSciBufObj), false);
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaTensorCreateFromNvSciBuf(device, *tensorSciBufObj, nvmTensor), false);
        
        return true;
    }

private:
    NvSciBufModule mNvSciBufModule{nullptr};
    std::vector<NvSciBufObj> mTensorSciBufObj;
};

#endif // ENABLE_DLA_API_API_3_0

#if ENABLE_DLA_API

class DlaContext
{
public:
    DlaContext() = default;

    bool createDla()
    {
        mDla = NvMediaDlaCreate();
        CHECK_RETURN_W_MSG(mDla, false, "Failed to create DLA");

        uint16_t numEngines{0};
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaGetNumEngines(mDla, &numEngines), false);
        CHECK_RETURN(numEngines > 1, false);

#if ENABLE_DLA_API_API_3_0
        uint32_t maxTasksLimit{0};
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaGetMaxOutstandingTasks(mDla, &maxTasksLimit), false);
        CHECK_RETURN(maxTasksLimit > 0, false);
#endif // ENABLE_DLA_API_API_3_0
        
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaInit(mDla, 1, 1), false);

        return true;
    }

#if ENABLE_DLA_API_API_3_0
    bool initNvSciBuf()
    {
        return mNvSciBuf.initialize();
    }
#endif // ENABLE_DLA_API_API_3_0

    void destroyDla()
    {
        if (mDla)
        {
            NvMediaDlaDestroy(mDla);
            mDla = nullptr;
        }
    }

    bool createDevice()
    {
        CHECK_RETURN_W_MSG(NvMediaDeviceCreate(), false, "Failed to create device");
        return true;
    }

    void destroyDevice()
    {
        if (nullptr != mDevice)
        {
            NvMediaDeviceDestroy(mDevice);
            mDevice = nullptr;
        }
    }

#if ENABLE_DLA_API_API_3_0
    bool createIoTensors()
    {
        auto createNvMediaTensor = [this](const NvMediaDlaTensorDescriptor& inDesc)
        {
            NvMediaTensor* nvmTensor{nullptr};
            mNvSciBuf.allocate(mDevice, inDesc, nvmTensor);
            return nvmTensor;
        };
        // Prepare I/O tensors
        for (unsigned i = 0, max = getNbInputTensors(); i < max; ++i)
        {
            NvMediaDlaTensorDescriptor inDesc;
            CHECK_RETURN(getInputDesc(inDesc, i), false);
            auto nvmTensor = createNvMediaTensor(inDesc);
            CHECK_RETURN(nvmTensor, false);
            mTensorIn.emplace_back(nvmTensor);
        }
        for (unsigned i = 0, max = getNbOutputTensors(); i < max; ++i)
        {
            NvMediaDlaTensorDescriptor outDesc;
            CHECK_RETURN(getOutputDesc(outDesc, i), false);
            auto nvmTensor = createNvMediaTensor(outDesc);
            CHECK_RETURN(nvmTensor, false);
            mTensorOut.emplace_back(nvmTensor);
        }
        return true;
    }

    uint32_t getNbInputTensors() const
    {
        int32_t nbInputs{0};
        NvMediaDlaGetNumOfInputTensors(mDla, &nbInputs);
        return nbInputs;
    }

    uint32_t getNbOutputTensors() const
    {
        int32_t nbOutputs{0};
        NvMediaDlaGetNumOfOutputTensors(mDla, &nbOutputs);
        return nbOutputs;
    }

    bool getOutputDesc(NvMediaDlaTensorDescriptor& desc, uint32_t idx) const
    {
        CHECK_RETURN(mDla, false);
        memset(&desc, 0, sizeof(desc));
        return (NVMEDIA_STATUS_OK == NvMediaDlaGetOutputTensorDescriptor(mDla, idx, &desc));
    }

    bool getInputDesc(NvMediaDlaTensorDescriptor& desc, uint32_t idx) const
    {
        CHECK_RETURN(mDla, false);
        memset(&desc, 0, sizeof(desc));
        return (NVMEDIA_STATUS_OK == NvMediaDlaGetInputTensorDescriptor(mDla, idx, &desc));
    }
#endif // ENABLE_DLA_API_API_3_0

    void destroyTensor()
    {
        for (auto& elem : mTensorIn)
        {
            NvMediaTensorDestroy(elem);
        }
        mTensorIn.clear();

        for (auto& elem : mTensorOut)
        {
            NvMediaTensorDestroy(elem);
        }
        mTensorOut.clear();
    }

    template <typename T>
    bool fillInputTensor(const std::vector<T>& inputBuf, const int index)
    {
        unsigned int inputBufSize = inputBuf.size() * sizeof(T);
        NvMediaTensor* nvmTensor = mTensorIn[index];
        NvMediaTensorSurfaceMap tensorMap{};
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaTensorLock(nvmTensor, NVMEDIA_TENSOR_ACCESS_WRITE, &tensorMap), false);
        CHECK_RETURN(tensorMap.size >= inputBufSize, false);
        std::memcpy(tensorMap.mapping, inputBuf.data(), inputBufSize);
        NvMediaTensorUnlock(nvmTensor);
        return true;
    }

    template <typename T>
    bool copyOutputTensor(std::vector<T>& outputBuf, const int index)
    {
        unsigned int outputBufSize = outputBuf.size() * sizeof(T);
        NvMediaTensor* nvmTensor = mTensorOut[index];
        NvMediaTensorSurfaceMap tensorMap{};
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaTensorLock(nvmTensor, NVMEDIA_TENSOR_ACCESS_READ, &tensorMap), false);
        CHECK_RETURN(tensorMap.size >= outputBufSize, false);
        std::memcpy(outputBuf.data(), tensorMap.mapping, outputBuf.size() * sizeof(T));
        NvMediaTensorUnlock(nvmTensor);
        return true;
    }

    bool submit()
    {
#if ENABLE_DLA_API_API_3_0
        // Prepare NvMediaDlaData
        std::vector<NvMediaDlaData> inputDlaData, outputDlaData;
        for (unsigned i = 0; i < getNbInputTensors(); ++i)
        {
            NvMediaDlaData data{NVMEDIA_DLA_DATA_TYPE_TENSOR, mTensorIn[i]};
            CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaDataRegister(mDla, &data, 0U), false);
            inputDlaData.emplace_back(std::move(data));
        }
        for (unsigned i = 0; i < getNbOutputTensors(); ++i)
        {
            NvMediaDlaData data{NVMEDIA_DLA_DATA_TYPE_TENSOR, mTensorOut[i]};
            CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaDataRegister(mDla, &data, 0U), false);
            outputDlaData.emplace_back(std::move(data));
        }

        NvMediaDlaArgs inputArgs{inputDlaData.data(), getNbInputTensors()};
        NvMediaDlaArgs outputArgs{outputDlaData.data(), getNbOutputTensors()};
        // Submit the task
        constexpr uint32_t taskTimeout = 30000; //!< in miliseconds
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaSubmit(mDla, &inputArgs, nullptr, &outputArgs, taskTimeout), false);

        gLogInfo << "DLA submit successfully" << std::endl;
        NvMediaTensorTaskStatus taskStatus = {};
        auto status = NvMediaTensorGetStatus(mTensorOut[0], NVMEDIA_TENSOR_TIMEOUT_INFINITE, &taskStatus);
        if (status != NVMEDIA_STATUS_OK)
        {
            gLogError << "NvMediaTensorGetStatus failed. status:" << status << std::endl;
            return false;
        }
        gLogInfo << "NvMediaTensorGetStatus successful" << std::endl;
        gLogInfo << "Operation duration: " << taskStatus.durationUs << " us" << std::endl;

        // Unregister DLA data
        auto unregisterDlaData = [this](std::vector<NvMediaDlaData>& vec)
        {
            std::for_each(vec.begin(), vec.end(), [this](NvMediaDlaData& d) { NvMediaDlaDataUnregister(mDla, &d); });
        };
        unregisterDlaData(inputDlaData);
        unregisterDlaData(outputDlaData);

        if (taskStatus.status != NVMEDIA_STATUS_OK)
        {
            gLogError << "Engine returned error." << std::endl;
            return false;
        }
#endif // ENABLE_DLA_API_API_3_0
        return true;
    }

    bool deserialize(void* data, size_t size)
    {
        CHECK_RETURN(isDLAHeader(static_cast<char*>(data)), false);
#if ENABLE_DLA_API_API_3_0
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaLoadableCreate(mDla, &mDlaLoadable), false);
        NvMediaDlaBinaryLoadable binary{static_cast<uint8_t*>(data), size};
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaAppendLoadable(mDla, binary, mDlaLoadable), false);
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaSetCurrentLoadable(mDla, mDlaLoadable), false);
        CHECK_RETURN(NVMEDIA_STATUS_OK == NvMediaDlaLoadLoadable(mDla), false);
#endif
        return true;
    }

    void destroy() noexcept
    {
        destroyTensor();
#if ENABLE_DLA_API_API_3_0
        mNvSciBuf.destroy();
        if (mDlaLoadable)
        {
            NvMediaDlaRemoveLoadable(mDla);
            NvMediaDlaLoadableDestroy(mDla, mDlaLoadable);
            mDlaLoadable = nullptr;
        }
#endif // ENABLE_DLA_API_API_3_0
        destroyDla();
        destroyDevice();
    }

    NvMediaDla* getDla()
    {
        return mDla;
    }

private:
    NvMediaDla* mDla{nullptr};
    NvMediaDevice* mDevice{nullptr};
    std::vector<NvMediaTensor*> mTensorIn;
    std::vector<NvMediaTensor*> mTensorOut;
#if ENABLE_DLA_API_API_3_0
    NvMediaDlaLoadable* mDlaLoadable{nullptr};
    NvSciBuf mNvSciBuf;
#endif // ENABLE_DLA_API_API_3_0
};
#endif

//!
//! \brief  The SampleNvMedia class implements the NvMedia sample.
//!
//! \details It creates the network using a single conv layer.
//!
class SampleNvMedia
{
public:
    SampleNvMedia(int batchSize) : mBatchSize(batchSize)
    {
        assert(mBatchSize >= 1);
    }

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
    bool teardown()
    {
        return true;
    }

    //!
    //! \brief Randomly intitializes buffer.
    //!
    template <typename T>
    void randomInit(std::vector<T>& buffer) const;

    //!
    //! \brief Verifies that the output.
    //!
    template <typename T>
    bool verifyOutput(const vector<T>& referenceBuf, const vector<T>& outputBuf) const;

protected:
    //!
    //! \brief Create a single layer Network and marks the output layers.
    //!
    void constructNetwork(nvinfer1::INetworkDefinition& network) const;

    //!
    //! \brief Explicitly set network I/O formats.
    //!
    void setNetworkIOFormats(INetworkDefinition& network, bool isInt8) const;

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network.

public:
    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    int mBatchSize{1};
};

//!
//! \brief Explicitly set network I/O formats.
//!
void SampleNvMedia::setNetworkIOFormats(INetworkDefinition& network, bool isInt8) const
{
    TensorFormat formats = isInt8 ? TensorFormat::kCHW32 : TensorFormat::kCHW16;
    DataType dataType = isInt8 ? DataType::kINT8 : DataType::kHALF;
    for (int i = 0, n = network.getNbInputs(); i < n; i++)
    {
        auto input = network.getInput(i);
        input->setType(dataType);
        input->setAllowedFormats(static_cast<TensorFormats>(1U << static_cast<int>(formats)));
    }

    for (int i = 0, n = network.getNbOutputs(); i < n; i++)
    {
        auto output = network.getOutput(i);
        output->setType(dataType);
        output->setAllowedFormats(static_cast<TensorFormats>(1U << static_cast<int>(formats)));
    }
}

//!
//! \brief Creates the network, configures the builder, and creates the network engine.
//!
//! \details This function creates a network and builds an engine to run in DLA safe mode.
//! The network consists of only one elementwise sum layer with FP16 precision.
//!
//! \return Returns true if the engine was created successfully and false otherwise.
//!
bool SampleNvMedia::build()
{
    auto builder = makeObjGuard<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    CHECK_RETURN(builder.get(), false);

    auto network = makeObjGuard<nvinfer1::INetworkDefinition>(builder->createNetwork());
    CHECK_RETURN(network.get(), false);

    auto config = makeObjGuard<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    CHECK_RETURN(config.get(), false);

    constructNetwork(*network);
    setNetworkIOFormats(*network, false);

    builder->setMaxBatchSize(mBatchSize);

    mEngine.reset();

    config->setMaxWorkspaceSize(256_MiB);
    config->setFlag(BuilderFlag::kFP16);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);

    samplesCommon::enableDLA(builder.get(), config.get(), 0);

    config->clearFlag(BuilderFlag::kGPU_FALLBACK);
    config->setEngineCapability(EngineCapability::kSAFE_DLA);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    CHECK_RETURN(mEngine.get(), false);

    mInputDims = network->getInput(0)->getDimensions();
    mOutputDims = network->getOutput(0)->getDimensions();

    return true;
}

//!
//! \brief Create the single layer Network and marks the output layers.
//!
void SampleNvMedia::constructNetwork(nvinfer1::INetworkDefinition& network) const
{
    nvinfer1::Dims inputDims{3, {32, 32, 32}, {}};

    auto inA = network.addInput("inputA", DataType::kHALF, inputDims);
    auto inB = network.addInput("inputB", DataType::kHALF, inputDims);

    auto layer = network.addElementWise(*inA, *inB, ElementWiseOperation::kSUM);
    ITensor* out = layer->getOutput(0);

    out->setName("output");
    network.markOutput(*out);
}

//!
//! \brief Runs the TensorRT inference engine for this sample.
//!
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleNvMedia::infer() const
{
#if ENABLE_DLA_API

    auto mem = makeObjGuard<nvinfer1::IHostMemory>(mEngine->serialize());
    CHECK_RETURN_W_MSG(mem.get(), false, "Engine serialization failed");

    auto context = makeObjGuard<DlaContext>(new DlaContext());

#if ENABLE_DLA_API_API_3_0
    CHECK_RETURN(context->initNvSciBuf(), false);
#endif // ENABLE_DLA_API_API_3_0
    CHECK_RETURN(context->createDevice(), false);
    CHECK_RETURN(context->createDla(), false);
    CHECK_RETURN(context->deserialize(mem->data(), mem->size()), false);

    const int inputBufSize = volume(mInputDims);
    const int expectedOutputSize = volume(mOutputDims);

    auto allocateVecHalf = [](int size) 
    {
        std::vector<half_float::half> vec(size);
        std::fill(vec.begin(), vec.end(), 0);
        return vec;
    };
    auto inputBufA = allocateVecHalf(inputBufSize);
    auto inputBufB = allocateVecHalf(inputBufSize);
    auto referenceBuf = allocateVecHalf(expectedOutputSize);
    auto outputBuf = allocateVecHalf(expectedOutputSize);

#if ENABLE_DLA_API_API_3_0
    CHECK_RETURN(context->createIoTensors(), false);
#endif

    randomInit(inputBufA);
    randomInit(inputBufB);

    CHECK_RETURN(context->fillInputTensor(inputBufA, 0), false);
    CHECK_RETURN(context->fillInputTensor(inputBufB, 1), false);

    CHECK_RETURN(context->submit(), false);

    CHECK_RETURN(context->copyOutputTensor(outputBuf, 0), false);
 
    std::transform(
        inputBufA.begin(), inputBufA.end(), inputBufB.begin(), referenceBuf.begin(), std::plus<half_float::half>());
    CHECK_RETURN(verifyOutput(referenceBuf, outputBuf), false);
#endif // ENABLE_DLA_API
    return true;
}

//!
//! \brief Randomly initializes buffer.
//!
template <typename T>
void SampleNvMedia::randomInit(std::vector<T>& buffer) const
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, 63);

    auto gen = [&dist, &mt]() { return T(dist(mt)); };
    std::generate(buffer.begin(), buffer.end(), gen);
}

//!
//! \brief Verifies that the output is correct and prints it.
//!
template <typename T>
bool SampleNvMedia::verifyOutput(const vector<T>& ref, const vector<T>& output) const
{
    return std::equal(ref.begin(), ref.end(), output.begin());
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

#if ENABLE_DLA_API
#if ENABLE_DLA_API_API_3_0
    gLogInfo << "Using NvMedia DLA API >= 3.0 that could support DLA safety runtime" << std::endl;
#else
    gLogError << "NvMedia DLA API < 3.0 is not supported" << std::endl;
    return gLogger.reportPass(sampleTest);
#endif
    constexpr int batchSize = 1;
    SampleNvMedia sample(batchSize);
    CHECK_RETURN(sample.build(), gLogger.reportFail(sampleTest));
    CHECK_RETURN(sample.infer(), gLogger.reportFail(sampleTest));
    CHECK_RETURN(sample.teardown(), gLogger.reportFail(sampleTest));
#else
    gLogError << "Unsupported platform, please make sure it is running on aarch64, QNX or android" << std::endl;
#endif

    return gLogger.reportPass(sampleTest);
}
