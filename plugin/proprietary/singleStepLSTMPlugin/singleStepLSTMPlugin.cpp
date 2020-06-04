/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include <cuda.h>

#if CUDA_VERSION >= 10000 && INCLUDE_MMA_KERNELS

#include "singleStepLSTMPlugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::SingleStepLSTMPlugin;
using nvinfer1::plugin::SingleStepLSTMPluginCreator;

REGISTER_TENSORRT_PLUGIN(SingleStepLSTMPluginCreator);

SingleStepLSTMPlugin::SingleStepLSTMPlugin(const PluginFieldCollection* fc)
{
    int idx = 0;

    mNumLayers = *(int*) (fc->fields[idx].data);
    idx++;

    mHiddenSize = *(int*) (fc->fields[idx].data);
    idx++;

    mAttentionSize = *(int*) (fc->fields[idx].data);
    idx++;

    mBeamSize = *(int*) (fc->fields[idx].data);
    idx++;

    mDataType = *(nvinfer1::DataType*) (fc->fields[idx].data);
    idx++;

    mDevice = -1;
    mSMVersionMajor = -1;
    mSMVersionMinor = -1;
}

SingleStepLSTMPlugin::SingleStepLSTMPlugin(const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data), *a = d;
    read<int>(d, mNumLayers);
    read<int>(d, mHiddenSize);
    read<int>(d, mAttentionSize);
    read<int>(d, mInputSize);
    read<int>(d, mBeamSize);
    read<int>(d, mDevice);
    read<int>(d, mSMVersionMajor);
    read<int>(d, mSMVersionMinor);

    read<nvinfer1::DataType>(d, mDataType);

    assert(d == a + length);
}

const char* SingleStepLSTMPlugin::getPluginType() const
{
    return "SingleStepLSTMPlugin";
}

const char* SingleStepLSTMPlugin::getPluginVersion() const
{
    return "1";
}

void SingleStepLSTMPlugin::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SingleStepLSTMPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

void SingleStepLSTMPlugin::destroy()
{
    delete this;
}

void SingleStepLSTMPlugin::setCUDAInfo(cudaStream_t mStreami, cudaStream_t mStreamh, cudaStream_t* mSplitKStreams,
    cudaEvent_t* mSplitKEvents, cublasHandle_t mCublas)
{
    this->mStreami = mStreami;
    this->mStreamh = mStreamh;
    this->mSplitKStreams = mSplitKStreams;
    this->mSplitKEvents = mSplitKEvents;
    this->mCublas = mCublas;
}

IPluginV2Ext* SingleStepLSTMPlugin::clone() const
{
    size_t sz = getSerializationSize();

    char* buff = (char*) malloc(getSerializationSize());

    serialize(buff);

    SingleStepLSTMPlugin* ret = new SingleStepLSTMPlugin(buff, sz);

    ret->setCUDAInfo(mStreami, mStreamh, mSplitKStreams, mSplitKEvents, mCublas);

    free(buff);
    ret->setPluginNamespace(mNamespace.c_str());
    return ret;
}

int SingleStepLSTMPlugin::getNbOutputs() const
{
    return 1 + 2 * mNumLayers;
}

// TODO: No idea if this needs batch size. Actually, don't know what's expected at all.
Dims SingleStepLSTMPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index >= 0 && index < this->getNbOutputs());

    // y/hy/cy are all hiddenSize * batch.
    return Dims3(inputs[0].d[0], 1, mHiddenSize);
}

// Only half for now
bool SingleStepLSTMPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return type == DataType::kHALF || type == DataType::kINT8;
}

void SingleStepLSTMPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    assert(*inputTypes == DataType::kHALF);
    mInputSize = inputDims[0].d[inputDims[0].nbDims - 1];
}

int SingleStepLSTMPlugin::initialize()
{
    CHECK(cublasCreate(&mCublas));

    CHECK(cublasSetMathMode(mCublas, CUBLAS_TENSOR_OP_MATH));

    CHECK(cudaStreamCreateWithPriority(&mStreami, 0, -1));
    CHECK(cudaStreamCreate(&mStreamh));
    mSplitKStreams = (cudaStream_t*) malloc(NUM_SPLIT_K_STREAMS * sizeof(cudaStream_t));
    mSplitKEvents = (cudaEvent_t*) malloc(NUM_SPLIT_K_STREAMS * sizeof(cudaEvent_t));

    for (int i = 0; i < NUM_SPLIT_K_STREAMS; i++)
    {
        CHECK(cudaStreamCreateWithPriority(&mSplitKStreams[i], 0, -1));
    }

    cudaError_t status;
    cudaDeviceProp deviceProperties;

    status = cudaGetDevice(&mDevice);
    if (status != cudaSuccess)
    {
        return status;
    }

    status = cudaGetDeviceProperties(&deviceProperties, mDevice);
    if (status != cudaSuccess)
    {
        return status;
    }
    mSMVersionMajor = deviceProperties.major;
    mSMVersionMinor = deviceProperties.minor;
    if (mSMVersionMajor <= 0 || mSMVersionMinor < 0)
    {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

void SingleStepLSTMPlugin::terminate()
{
    if (mCublas)
    {
        CHECK(cublasDestroy(mCublas));
        mCublas = nullptr;
    }

    if (mStreami)
    {
        CHECK(cudaStreamDestroy(mStreami));
        mStreami = nullptr;
    }
    if (mStreamh)
    {
        CHECK(cudaStreamDestroy(mStreamh));
        mStreamh = nullptr;
    }

    for (int i = 0; i < NUM_SPLIT_K_STREAMS; i++)
    {
        if (mSplitKStreams[i])
        {
            CHECK(cudaStreamDestroy(mSplitKStreams[i]));
            mSplitKStreams[i] = nullptr;
        }
    }

    if (mSplitKStreams)
    {
        free(mSplitKStreams);
        mSplitKStreams = nullptr;
    }
    if (mSplitKEvents)
    {
        free(mSplitKEvents);
        mSplitKEvents = nullptr;
    }
}

size_t SingleStepLSTMPlugin::getWorkspaceSize(int maxBatchSize) const
{
    size_t size = 0;

    // tmp_io
    size += mNumLayers * (mAttentionSize + mInputSize) * maxBatchSize * mBeamSize * sizeof(half);

    // tmp_i
    size += mHiddenSize * maxBatchSize * mBeamSize * 4 * NUM_SPLIT_K_STREAMS * sizeof(half);

    // tmp_h
    size += mNumLayers * mHiddenSize * maxBatchSize * mBeamSize * 4 * sizeof(half);

    return size;
}

int SingleStepLSTMPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    int effectiveBatch = batchSize * mBeamSize;

    assert(mAttentionSize == mHiddenSize);
    assert(mInputSize == mHiddenSize);

    void* tmp_io = workspace;
    void* tmp_i
        = (void*) ((char*) (workspace) + mNumLayers * (mAttentionSize + mInputSize) * effectiveBatch * sizeof(half));
    void* tmp_h = (void*) ((char*) (tmp_i) + mHiddenSize * effectiveBatch * 4 * NUM_SPLIT_K_STREAMS * sizeof(half));

    cudaEvent_t event;
    CHECK(cudaEventCreate(&event, cudaEventDisableTiming));
    CHECK(cudaEventRecord(event, stream));
    CHECK(cudaStreamWaitEvent(mStreami, event, 0));
    CHECK(cudaStreamWaitEvent(mStreamh, event, 0));
    for (int i = 0; i < NUM_SPLIT_K_STREAMS; i++)
    {
        CHECK(cudaStreamWaitEvent(mSplitKStreams[i], event, 0));
    }
    CHECK(cudaEventDestroy(event));

    cudaError_t status;
    int device;
    cudaDeviceProp deviceProperties;

    status = cudaGetDevice(&device);
    if (status != cudaSuccess)
    {
        return status;
    }
    assert(device == mDevice);

    if (mSMVersionMajor <= 0 || mSMVersionMinor < 0)
    {
        status = cudaGetDeviceProperties(&deviceProperties, mDevice);
        if (status != cudaSuccess)
        {
            return status;
        }
        mSMVersionMajor = deviceProperties.major;
        mSMVersionMinor = deviceProperties.minor;
        if (mSMVersionMajor <= 0 || mSMVersionMinor < 0)
        {
            return cudaErrorUnknown;
        }
    }

    int inputSize = mInputSize + mAttentionSize;

    // The SM version needs to be 7.5 and mHiddenSize needs to be a multiple of 8 to run small Gemm kernel.
    // The first small Gemm needs 128-bit alignment for x, w and tmp_io, and 32-bit alignment for tmp_i.
    // Second small Gemm needs 128-bit alignment for hx, w, and 16-bit alignment for tmp_h;
    bool smallGemm = (mSMVersionMajor == 7) && (mSMVersionMinor == 5) && (mHiddenSize % 128 == 0);
    bool firstSmallGemm = smallGemm && ((uintptr_t) inputs[0] % 16 == 0) && ((uintptr_t) tmp_io % 16 == 0)
        && ((uintptr_t) tmp_i % 2 == 0);
    bool secondSmallGemm = smallGemm && ((uintptr_t) tmp_h % 2 == 0);

    for (int i = 0; i < mNumLayers; i++)
    {
        firstSmallGemm = firstSmallGemm && ((uintptr_t) inputs[2 + 2 * mNumLayers + i] % 16 == 0);
        secondSmallGemm = secondSmallGemm && ((uintptr_t) inputs[2 + 2 * mNumLayers + i] % 16 == 0)
            && ((uintptr_t) inputs[2 + i] % 16 == 0);
    }

    if (mDataType == nvinfer1::DataType::kHALF)
    {
        using kernel = void (*)(int hiddenSize, int inputSize, int miniBatch, int seqLength, int numLayers,
            cublasHandle_t cublasHandle, half* x, half** hx, half** cx, half** w, half** bias, half* y, half** hy,
            half** cy, half* concatData, half* tmp_io, half* tmp_i, half* tmp_h,
#if (BATCHED_GEMM)
            half** aPtrs, half** bPtrs, half** cPtrs,
#endif
            cudaStream_t streami, cudaStream_t* splitKStreams, cudaEvent_t* splitKEvents, int numSplitKStreams,
            cudaStream_t streamh);
        kernel funcs[] = {singleStepLSTMKernel<CUDA_R_16F, CUDA_R_16F, false, false>,
            singleStepLSTMKernel<CUDA_R_16F, CUDA_R_16F, false, true>,
            singleStepLSTMKernel<CUDA_R_16F, CUDA_R_16F, true, false>,
            singleStepLSTMKernel<CUDA_R_16F, CUDA_R_16F, true, true>};

        // Every variable (firstSmallGemm, secondSmallGemm) has 2 options, which correspond 2 kernels when other
        // variables are the same. int(firstSmallGemm) * 2 + int(secondSmallGemm) is getting the correct index of the
        // specific kernel.
        kernel func = funcs[int(firstSmallGemm) * 2 + int(secondSmallGemm)];

        func(mHiddenSize, inputSize, effectiveBatch, 1, mNumLayers, this->mCublas,
            (half*) inputs[0],                      // x
            (half**) (&(inputs[2])),                // Array of hx,
            (half**) (&inputs[2 + mNumLayers]),     // Array of cx,
            (half**) (&inputs[2 + 2 * mNumLayers]), // w,
            (half**) (&inputs[2 + 3 * mNumLayers]), // bias
            (half*) outputs[0],                     // y,
            (half**) (&outputs[1]),                 // Array of hy,
            (half**) (&outputs[1 + mNumLayers]),    // Array of cy,
            (half*) inputs[1],                      // concatData,
            (half*) tmp_io, (half*) tmp_i, (half*) tmp_h, mStreami, mSplitKStreams, mSplitKEvents, NUM_SPLIT_K_STREAMS,
            mStreamh);
    }

    cudaEvent_t eventEnd;

    // The final kernel is the elementwise kernel launched to stream i, so only need to wait for that one to finish.
    CHECK(cudaEventCreate(&eventEnd, cudaEventDisableTiming));
    CHECK(cudaEventRecord(eventEnd, mStreami));
    CHECK(cudaStreamWaitEvent(stream, eventEnd, 0));
    CHECK(cudaEventDestroy(eventEnd));

    return 0;
}

size_t SingleStepLSTMPlugin::getSerializationSize() const
{
    size_t sz = sizeof(mNumLayers) + sizeof(mHiddenSize) + sizeof(mAttentionSize) + sizeof(mInputSize)
        + sizeof(mBeamSize) + sizeof(mDataType) + sizeof(mSMVersionMajor) + sizeof(mSMVersionMinor) + sizeof(mDevice);
    return sz;
}

void SingleStepLSTMPlugin::serialize(void* buffer) const
{
    char *d = static_cast<char*>(buffer), *a = d;

    write<int>(d, mNumLayers);
    write<int>(d, mHiddenSize);
    write<int>(d, mAttentionSize);
    write<int>(d, mInputSize);
    write<int>(d, mBeamSize);
    write<int>(d, mDevice);
    write<int>(d, mSMVersionMajor);
    write<int>(d, mSMVersionMinor);
    write<nvinfer1::DataType>(d, mDataType);

    assert(d == a + getSerializationSize());
}

nvinfer1::DataType SingleStepLSTMPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return mDataType;
}

bool SingleStepLSTMPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool SingleStepLSTMPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return inputIndex >= 2 * mNumLayers + 2;
}

template <typename T>
void SingleStepLSTMPlugin::write(char*& buffer, const T& val) const
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void SingleStepLSTMPlugin::read(const char*& buffer, T& val) const
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}

const char* SingleStepLSTMPluginCreator::getPluginName() const
{
    return "SingleStepLSTMPlugin";
}

const char* SingleStepLSTMPluginCreator::getPluginVersion() const
{
    return "1";
}

// Not sure why I need names. Can't do it with variable layer count anyway.
const PluginFieldCollection* SingleStepLSTMPluginCreator::getFieldNames()
{
    return nullptr;
}

void SingleStepLSTMPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* SingleStepLSTMPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

IPluginV2Ext* SingleStepLSTMPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    return new SingleStepLSTMPlugin(fc);
}

IPluginV2Ext* SingleStepLSTMPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    return new SingleStepLSTMPlugin(serialData, serialLength);
}

#endif /* CUDA_VERSION >= 10000 && INCLUDE_MMA_KERNELS */
