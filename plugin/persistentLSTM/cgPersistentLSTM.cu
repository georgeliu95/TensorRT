#ifdef __linux__
#ifdef __x86_64__

#include <string>
#include <sstream>
#include <dlfcn.h>
#include <iostream>
#include "cgPersistentLSTM.h"
#include "cgPersistentLSTMKernel.h"
#include "cgPersistentLSTMPlugin.h"
#include "checkMacros.h"
#include "helpers.h"
#include "cudaDriverWrapper.h"

using namespace nvinfer1;
using nvinfer1::plugin::CgPersistentLSTM;
using nvinfer1::plugin::CgPersistentLSTMPluginCreator;
using nvinfer1::plugin::CgPersistentLSTMPlugin;

// There seems to be a linking problem. So I just copied nse.cu for now
#define MAX_THREADS_PER_EMBEDDING 128
#define THREADS(E) std::min(MAX_THREADS_PER_EMBEDDING, E)

#define cuErrCheck(stat, wrap)                                                                                               \
    {                                                                                                                  \
        cuErrCheck_((stat), wrap, __FILE__, __LINE__);                                                                       \
    }
void cuErrCheck_(CUresult stat, const CUDADriverWrapper & wrap, const char* file, int line)
{
    if (stat != CUDA_SUCCESS)
    {
        const char* msg;
        wrap.cuGetErrorName(stat, &msg);
        fprintf(stderr, "CUDA Error: %s %s %d\n", msg, file, line);
    }
}

#define nvrtcErrCheck(stat)                                                                                            \
    {                                                                                                                  \
        nvrtcErrCheck_((stat), __FILE__, __LINE__);                                                                    \
    }
void nvrtcErrCheck_(nvrtcResult stat, const char* file, int line)
{
    if (stat != NVRTC_SUCCESS)
    {
        fprintf(stderr, "nvrtc Error: %d %s %d\n", stat, file, line);
    }
}

#define CUBLASERRORMSG(status_)                                                 \
    {                                                                         \
        auto s_ = status_;                                                    \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                      \
        {                                                                     \
            nvinfer1::logError(#status_ " failure.", __FILE__, FN_NAME, __LINE__);      \
        }                                                                     \
    }

void nvrtcCompile(CUlinkState linker, nvrtcProgram* prog, const char* src, const char* fileName, const char* funcName,
    const char** loweredName, const int numOpts, const char** opts, const CUDADriverWrapper &wrap)
{
    nvrtcErrCheck(nvrtcCreateProgram(prog, src, fileName, 0, nullptr, nullptr));

    nvrtcErrCheck(nvrtcAddNameExpression(*prog, funcName));

    nvrtcResult compileResult = nvrtcCompileProgram(*prog, numOpts, opts);

    if (compileResult != NVRTC_SUCCESS)
    {
        size_t logSize;
        nvrtcErrCheck(nvrtcGetProgramLogSize(*prog, &logSize));
        char* log = new char[logSize];
        nvrtcErrCheck(nvrtcGetProgramLog(*prog, log));
        printf("NVRTC compile error\n");
        printf("%s\n", log);
        delete[] log;
        return;
    }

    size_t ptxSize;
    nvrtcErrCheck(nvrtcGetPTXSize(*prog, &ptxSize));

    std::vector<char> ptx(ptxSize);
    nvrtcErrCheck(nvrtcGetPTX(*prog, ptx.data()));

    nvrtcErrCheck(nvrtcGetLoweredName(*prog, funcName, loweredName));

    cuErrCheck(wrap.cuLinkAddData(linker, CU_JIT_INPUT_PTX, ptx.data(), ptxSize, funcName, 0, nullptr, nullptr), wrap);

}

inline __device__ int32_t offset(int32_t row, int32_t stride, int32_t col) { return row * stride + col; }

template <typename T>
__global__ __launch_bounds__(MAX_THREADS_PER_EMBEDDING) void nse2sneV(const void* __restrict__ _nse,
    void* __restrict__ _sne, const int32_t* lengths, const int32_t* permutation, int32_t E)
{
    const T* nse = static_cast<const T*>(_nse);
    T** sne = static_cast<T**>(_sne);
    if (lengths[blockIdx.x] > blockIdx.y)
    {
        for (int32_t i = threadIdx.x; i < E; i += blockDim.x)
        {
            sne[blockIdx.y][permutation[blockIdx.x] * E + i] = nse[offset(blockIdx.x, gridDim.y, blockIdx.y) * E + i];
        }
    }

}

template <typename T>
__global__ __launch_bounds__(MAX_THREADS_PER_EMBEDDING) void sne2nseV(const void* __restrict__ _sne,
    void* __restrict__ _nse, const int32_t* lengths, const int32_t* permutation, int32_t E)
{
    const T* const* sne = static_cast<const T* const*>(_sne);
    T* nse = static_cast<T*>(_nse);
    for (int32_t i = threadIdx.x; i < E; i += blockDim.x)
    {
        T val = lengths[blockIdx.x] > blockIdx.y ? sne[blockIdx.y][permutation[blockIdx.x] * E + i] : T();
        nse[offset(blockIdx.x, gridDim.y, blockIdx.y) * E + i] = val;
    }
}

template <typename T>
__global__ __launch_bounds__(MAX_THREADS_PER_EMBEDDING) void nse2sne(
    const void* __restrict__ _nse, void* __restrict__ _sne, const int32_t* permutation, int32_t E)
{
    const T* nse = static_cast<const T*>(_nse);
    T* sne = static_cast<T*>(_sne);
    for (int32_t i = threadIdx.x; i < E; i += blockDim.x)
    {
        sne[offset(blockIdx.y, gridDim.x, permutation[blockIdx.x]) * E + i]
            = nse[offset(blockIdx.x, gridDim.y, blockIdx.y) * E + i];
    }

}

template <typename T>
__global__ __launch_bounds__(MAX_THREADS_PER_EMBEDDING) void nse2sneB(
    const void* __restrict__ _nse, void* __restrict__ _sne, const int32_t* permutation, int32_t E)
{
    const T* nse = static_cast<const T*>(_nse);
    T* sne = static_cast<T*>(_sne);

    int32_t nseIdx = offset(blockIdx.x, gridDim.y, blockIdx.y) * 2 * E;
    int32_t sneIdx = offset(blockIdx.y, gridDim.x * 2, permutation[blockIdx.x]) * E;
    for (int32_t i = threadIdx.x; i < E; i += blockDim.x)
        sne[sneIdx + i] = nse[nseIdx + i];

    nseIdx += E;
    sneIdx += gridDim.x * E;
    for (int32_t i = threadIdx.x; i < E; i += blockDim.x)
    {
        sne[sneIdx + i] = nse[nseIdx + i];
    }
}

template <typename T>
__global__ __launch_bounds__(MAX_THREADS_PER_EMBEDDING) void sne2nse(
    const void* __restrict__ _sne, void* __restrict__ _nse, const int32_t* permutation, int32_t E)
{
    const T* sne = static_cast<const T*>(_sne);
    T* nse = static_cast<T*>(_nse);
    for (int32_t i = threadIdx.x; i < E; i += blockDim.x)
    {
        nse[offset(blockIdx.x, gridDim.y, blockIdx.y) * E + i]
            = sne[offset(blockIdx.y, gridDim.x, permutation[blockIdx.x]) * E + i];
    }
}

template <typename T>
__global__ __launch_bounds__(MAX_THREADS_PER_EMBEDDING) void sne2nseB(
    const void* __restrict__ _sne, void* __restrict__ _nse, const int32_t* permutation, int32_t E)
{
    T* nse = static_cast<T*>(_nse);
    const T* sne = static_cast<const T*>(_sne);

    int32_t nseIdx = offset(blockIdx.x, gridDim.y, blockIdx.y) * 2 * E;
    int32_t sneIdx = offset(blockIdx.y, gridDim.x * 2, permutation[blockIdx.x]) * E;
    for (int32_t i = threadIdx.x; i < E; i += blockDim.x)
    {
        nse[nseIdx + i] = sne[sneIdx + i];
    }

    nseIdx += E;
    sneIdx += gridDim.x * E;
    for (int32_t i = threadIdx.x; i < E; i += blockDim.x)
    {
        nse[nseIdx + i] = sne[sneIdx + i];
    }
}

typedef void (*vconv_t)(const void*, void*, const int32_t*, const int32_t*, int32_t);
typedef void (*conv_t)(const void*, void*, const int32_t*, int32_t);

vconv_t conversionV[][2] = {{sne2nseV<float>, nse2sneV<float>}, {sne2nseV<half>, nse2sneV<half>},
    {sne2nseV<int8_t>, nse2sneV<int8_t>}, {sne2nseV<int32_t>, nse2sneV<int32_t>}};

conv_t conversion[][2] = {{sne2nse<float>, nse2sne<float>}, {sne2nse<half>, nse2sne<half>},
    {sne2nse<int8_t>, nse2sne<int8_t>}, {sne2nse<int32_t>, nse2sne<int32_t>}};

conv_t conversionB[][2] = {{sne2nseB<float>, nse2sneB<float>}, {sne2nseB<half>, nse2sneB<half>},
    {sne2nseB<int8_t>, nse2sneB<int8_t>}, {sne2nseB<int32_t>, nse2sneB<int32_t>}};

void reformatInternal(bool toSNE, DataType type, bool bi, const void* src, void* dst, const int* lengths,
    const int* permutation, int nbSeq, int maxSeqLength, int E, cudaStream_t stream)
{
    dim3 NxS(nbSeq, maxSeqLength);
    if (lengths)
        conversionV[static_cast<int32_t>(type)][toSNE]<<<NxS, THREADS(E), 0, stream>>>(
            src, dst, lengths, permutation, E);
    else
    {
        if (bi)
            conversionB[static_cast<int32_t>(type)][toSNE]<<<NxS, THREADS(E), 0, stream>>>(src, dst, permutation, E);
        else
            conversion[static_cast<int32_t>(type)][toSNE]<<<NxS, THREADS(E), 0, stream>>>(src, dst, permutation, E);
    }
    CUASSERT(cudaGetLastError());
}

void* reformatRNNForwardInferenceTensor2SNE(DataType type, bool bi, const void* src, void* dst, const int* lengths,
    const int32_t* permutation, int nbSeq, int maxSeqLength, int E, cudaStream_t stream)
{
    reformatInternal(true, type, bi, src, dst, lengths, permutation, nbSeq, maxSeqLength, E, stream);
    return dst;
}

void* reformatRNNForwardInferenceTensor2NSE(DataType type, bool bi, const void* src, void* dst, const int32_t* lengths,
    const int32_t* permutation, int32_t nbSeq, int32_t maxSeqLength, int32_t E, cudaStream_t stream)
{
    reformatInternal(false, type, bi, src, dst, lengths, permutation, nbSeq, maxSeqLength, E, stream);
    return dst;
}

int roundUp(int a, int multiple) { return a + (a % multiple ? multiple - a % multiple : 0); }

// Copied from cudaRNNBaseRunner
size_t rounded(size_t sz)
{
    // Round up to multiple of 8 to help alignment for largest type
    return roundUp(sz, 8);
}

CgPersistentLSTM::CgPersistentLSTM(
    int batchSize, int seqLen, int inputSize, nvinfer1::DataType dataType, CgPLSTMParameters param)
    : param(param)
    , maxBatchSize(batchSize)
    , maxSeqLen(seqLen)
    , dataType(dataType)
    , inputSize(inputSize)
{

    dirMul = param.isBi ? 2 : 1;
    dataSize = dataType == nvinfer1::DataType::kFLOAT ? 4 : 2;
    d_miniBatchArray = nullptr;
    hostScratch = nullptr;

    CUBLASASSERT(cublasCreate(&handle));
    CUBLASASSERT(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    setupScratchSizes(); // Device scratch space

    setupPersistentSizes(); // Host scrath space
}


void CgPersistentLSTM::setupScratchSizes()
{
    // cgPLSTMworkspace
    int tmpIspace = dirMul * 4 * maxBatchSize * maxSeqLen * param.hiddenSize * dataSize; // working with tensorcore now

    int tmpHspace = 4 * roundUp(param.hiddenSize, param.FRAG_K) * roundUp(maxBatchSize, param.FRAG_N)
        * param.blockSplitKFactor * dataSize;
    if (!param.separatePath)
    {
        tmpHspace *= dirMul;
    }
    mScratch.tmpI.size = rounded(tmpIspace);
    mScratch.tmpH.size = rounded(tmpHspace);
    mScratch.permutation.size = rounded(sizeof(int32_t) * maxBatchSize);
    mScratch.lengths.size = rounded(sizeof(int32_t) * maxBatchSize);
    // X offsets
    mScratch.x.size = rounded(sizeof(void*) * maxSeqLen);
    // Y offsets
    mScratch.y.size = rounded(sizeof(void*) * maxSeqLen);
    // Inputs
    mScratch.ix.size = rounded(dataSize * maxBatchSize * maxSeqLen * inputSize);
    mScratch.hx.size = rounded(dirMul * dataSize * maxBatchSize * param.numLayers * param.hiddenSize);
    mScratch.cx.size = rounded(dirMul * dataSize * maxBatchSize * param.numLayers * param.hiddenSize);

    // Outputs
    mScratch.oy.size = rounded(dirMul * dataSize * maxBatchSize * maxSeqLen * param.hiddenSize);
    mScratch.hy.size = rounded(dirMul * dataSize * maxBatchSize * param.numLayers * param.hiddenSize);
    mScratch.cy.size = rounded(dirMul * dataSize * maxBatchSize * param.numLayers * param.hiddenSize);
}

void CgPersistentLSTM::setupPersistentSizes()
{
    mPersistentHost.permutation.size = rounded(sizeof(int32_t) * maxBatchSize);
    mPersistentHost.lengths.size = rounded(sizeof(int32_t) * maxBatchSize);

    mPersistentHost.x.size = rounded(sizeof(void*) * maxSeqLen);
    mPersistentHost.y.size = rounded(sizeof(void*) * maxSeqLen);
    mPersistentHost.validBatch.size = rounded(sizeof(int32_t) * maxSeqLen);
}

size_t CgPersistentLSTM::computeScratchSize() const
{
    return mScratch.tmpI.size + mScratch.tmpH.size + mScratch.permutation.size + mScratch.lengths.size + mScratch.x.size
        + mScratch.y.size + mScratch.ix.size + mScratch.oy.size + mScratch.hx.size + mScratch.cx.size + mScratch.hy.size
        + mScratch.cy.size;
}

size_t CgPersistentLSTM::computeGpuSize(
    int maxBatchSize, int maxSeqLen, int inputSize, int dataSize, CgPLSTMParameters param)
{

    int dirMul = param.isBi ? 2 : 1;
    int tmpIspace
        = dirMul * 4 * maxBatchSize * maxSeqLen * param.hiddenSize * dataSize; // working with tensorcore now

    int tmpHspace = 4 * roundUp(param.hiddenSize, param.FRAG_K) * roundUp(maxBatchSize, param.FRAG_N)
        * param.blockSplitKFactor * dataSize;

    if (!param.separatePath)
    {
        tmpHspace *= dirMul;
    }

    size_t retSize = 0;
    retSize += rounded(tmpIspace);
    retSize += rounded(tmpHspace);
    retSize += rounded(sizeof(int32_t) * maxBatchSize);
    retSize += rounded(sizeof(int32_t) * maxBatchSize);
    // X offsets
    retSize += rounded(sizeof(void*) * maxSeqLen);
    // Y offsets
    retSize += rounded(sizeof(void*) * maxSeqLen);
    // Inputs
    retSize += rounded(dataSize * maxBatchSize * maxSeqLen * inputSize);
    retSize += rounded(dirMul * dataSize * maxBatchSize * param.numLayers * param.hiddenSize);
    retSize += rounded(dirMul * dataSize * maxBatchSize * param.numLayers * param.hiddenSize);

    // Outputs
    retSize += rounded(dirMul * dataSize * maxBatchSize * maxSeqLen * param.hiddenSize);
    retSize += rounded(dirMul * dataSize * maxBatchSize * param.numLayers * param.hiddenSize);
    retSize += rounded(dirMul * dataSize * maxBatchSize * param.numLayers * param.hiddenSize);

    return retSize;
}

size_t CgPersistentLSTM::computePersistentHostSize() const
{
    return mPersistentHost.permutation.size + mPersistentHost.lengths.size + mPersistentHost.x.size
        + mPersistentHost.y.size + mPersistentHost.validBatch.size;
}

void CgPersistentLSTM::setupPointers(void* deviceScratch)
{
    setupPersistentSizes();
    size_t hostScratchSize = computePersistentHostSize();

    hostScratch = new char[hostScratchSize];
    char* hostScratchTemp = hostScratch;

    auto splitOut = [](void*& p, SizedRegion& region) {
        region.data = p;
        p = static_cast<char*>(p) + region.size;
    };

    auto splitOutChar = [](char*& p, SizedRegion& region) {
        region.data = p;
        p = p + region.size;
    };

    splitOut(deviceScratch, mScratch.tmpI);
    splitOut(deviceScratch, mScratch.tmpH);
    splitOut(deviceScratch, mScratch.permutation);
    splitOut(deviceScratch, mScratch.lengths);
    splitOut(deviceScratch, mScratch.x);
    splitOut(deviceScratch, mScratch.y);
    splitOut(deviceScratch, mScratch.ix);
    splitOut(deviceScratch, mScratch.hx);
    splitOut(deviceScratch, mScratch.cx);
    splitOut(deviceScratch, mScratch.oy);
    splitOut(deviceScratch, mScratch.hy);
    splitOut(deviceScratch, mScratch.cy);

    splitOutChar(hostScratchTemp, mPersistentHost.permutation);
    splitOutChar(hostScratchTemp, mPersistentHost.lengths);
    splitOutChar(hostScratchTemp, mPersistentHost.x);
    splitOutChar(hostScratchTemp, mPersistentHost.y);
    splitOutChar(hostScratchTemp, mPersistentHost.validBatch);
}

// mostly from cudaRNNBaseRunner.cpp
void CgPersistentLSTM::doInputTranspose(const void* x, void* y, const void* hx, const void* cx, void* hy, void* cy,
    const void* sequenceLengths, int inputBatchSize, cudaStream_t stream)
{
    cpuLengths = static_cast<int32_t*>(mPersistentHost.lengths.data);
    setupSequenceLengths(cpuLengths, inputBatchSize, sequenceLengths, stream);
    effectiveSeqLen = *std::max_element(cpuLengths, cpuLengths + inputBatchSize);

    cpuPermutation = static_cast<int32_t*>(mPersistentHost.permutation.data);
    std::vector<int32_t> sne_offset(maxSeqLen);

    setupSNEPermutation(cpuLengths, cpuPermutation, inputBatchSize);
    setupSNEOffsetsWidths(cpuLengths, sne_offset.data(), static_cast<int32_t*>(mPersistentHost.validBatch.data),
        inputBatchSize, maxSeqLen);
    for (int i = effectiveSeqLen; i < maxSeqLen; i++)
    {
        static_cast<int*>(mPersistentHost.validBatch.data)[i] = 0;
    }

    bool doPermutation = !std::is_sorted(cpuPermutation, cpuPermutation + inputBatchSize);

    NZSE2SNZE = inputBatchSize > 1 && maxSeqLen > 1;

    NZLH2LNZH = (param.isBi && (inputBatchSize > 1 || param.numLayers > 1 || param.hiddenSize > 1))
        || (!param.isBi && (inputBatchSize > 1 && param.numLayers > 1)) || doPermutation;
    if (NZSE2SNZE || NZLH2LNZH)
    {
        gpuPermutation = static_cast<int32_t*>(mScratch.permutation.data);
        CUASSERT(cudaMemcpyAsync(
            gpuPermutation, cpuPermutation, sizeof(int32_t) * inputBatchSize, cudaMemcpyHostToDevice, stream));
    }

    if (NZSE2SNZE)
    {
        gpuX = mScratch.x.data;
        gpuY = mScratch.y.data;
        gpuLengths = static_cast<int32_t*>(mScratch.lengths.data);

        void** sneXStart = static_cast<void**>(mPersistentHost.x.data);
        void** sneYStart = static_cast<void**>(mPersistentHost.y.data);
        setupSNEOffsetsGPU(sne_offset.data(), sneXStart, mScratch.ix.data, maxSeqLen, inputSize * dataSize);
        setupSNEOffsetsGPU(
            sne_offset.data(), sneYStart, mScratch.oy.data, maxSeqLen, param.hiddenSize * dirMul * dataSize);

        CUASSERT(cudaMemcpyAsync(gpuX, sneXStart, sizeof(void*) * maxSeqLen, cudaMemcpyHostToDevice, stream));
        CUASSERT(cudaMemcpyAsync(gpuY, sneYStart, sizeof(void*) * maxSeqLen, cudaMemcpyHostToDevice, stream));
        CUASSERT(
            cudaMemcpyAsync(gpuLengths, cpuLengths, sizeof(int32_t) * inputBatchSize, cudaMemcpyHostToDevice, stream));

        reformatRNNForwardInferenceTensor2SNE(
            dataType, false, x, gpuX, gpuLengths, gpuPermutation, inputBatchSize, maxSeqLen, inputSize, stream);
        xT = sneXStart[0];
        yT = sneYStart[0];
    }
    else if (std::any_of(cpuLengths, cpuLengths + inputBatchSize, [&](int32_t l) {
                 return l != maxSeqLen;
             })) // When padding is needed the tensor must be cleared if skipping transpose
    {
        int ySize = inputBatchSize * maxSeqLen * param.hiddenSize * dirMul * dataSize;
        CUASSERT(cudaMemsetAsync(y, 0, ySize, stream));
    }

    if (!NZSE2SNZE)
    {
        xT = x;
        yT = y;
    }

    if (NZLH2LNZH)
    {
        if (hx)
        {
            void* gpuHiddenInput = mScratch.hx.data;
            hxT = reformatRNNForwardInferenceTensor2SNE(dataType, param.isBi, hx, gpuHiddenInput, nullptr,
                    gpuPermutation, inputBatchSize, param.numLayers, param.hiddenSize, stream);
        }
        else
        {
            CUASSERT(cudaMemsetAsync(mScratch.hy.data, 0, mScratch.hx.size, stream));
            hxT = mScratch.hx.data;

        }

        if (cx)
        {
            void* gpuCellInput = mScratch.cx.data;
            cxT = reformatRNNForwardInferenceTensor2SNE(dataType, param.isBi, cx, gpuCellInput, nullptr, gpuPermutation,
                    inputBatchSize, param.numLayers, param.hiddenSize, stream);
        }
        else
        {
            CUASSERT(cudaMemsetAsync(mScratch.cx.data, 0,  mScratch.cx.size, stream));
            cxT = mScratch.cx.data;
        }
        if (hy)
        {
            hyT = mScratch.hy.data;
        }
        if (cy)
        {
            cyT = mScratch.cy.data;
        }
    }

    if (!NZLH2LNZH)
    {
        if (param.setInitialStates)
        {
            hxT = hx;
            cxT = cx;
        }
        else
        {
            CUASSERT(cudaMemsetAsync(hy, 0, mScratch.hx.size, stream));
            hxT = mScratch.hx.data;
            CUASSERT(cudaMemsetAsync(mScratch.cx.data, 0,  mScratch.cx.size, stream));
            cxT = mScratch.cx.data;
        }

        hyT = hy;
        cyT = cy;
    }
}

void CgPersistentLSTM::doOutputTranspose(void* y, void* hy, void* cy, int inputBatchSize, cudaStream_t stream)
{
    if (NZSE2SNZE)
    {
        reformatRNNForwardInferenceTensor2NSE(dataType, false, gpuY, y, gpuLengths, gpuPermutation, inputBatchSize,
            maxSeqLen, param.hiddenSize * dirMul, stream);
    }
    if (NZLH2LNZH)
    {
        if (hy)
        {
            reformatRNNForwardInferenceTensor2NSE(dataType, param.isBi, hyT, hy, nullptr, gpuPermutation,
                inputBatchSize, param.numLayers, param.hiddenSize, stream);
        }
        if (cy)
        {
            reformatRNNForwardInferenceTensor2NSE(dataType, param.isBi, cyT, cy, nullptr, gpuPermutation,
                inputBatchSize, param.numLayers, param.hiddenSize, stream);
        }
    }
}

void CgPersistentLSTM::setupSequenceLengths(
    int32_t* lengths, int inputBatchSize, const void* sequenceLengths, cudaStream_t stream)
{
    size_t seqLenNBytes = sizeof(int32_t) * inputBatchSize;
    CUASSERT(cudaMemcpyAsync(lengths, sequenceLengths, seqLenNBytes, cudaMemcpyDeviceToHost, stream));
    CUASSERT(cudaStreamSynchronize(stream));

    for (int i = 0; i < inputBatchSize; i++)
    {
        if (lengths[i] < 1 || maxSeqLen < lengths[i])
        {
            throwMiscError(__FILE__, FN_NAME, __LINE__, 1,
                "Input sequence lengths tensor contains a value that is outside the range [1, maxSeqLen]");
        }
    }
}

void CgPersistentLSTM::execute(const void* x, void* y, const void* init_h, const void* init_c, void* final_h,
    void* final_c, const void* rMat, const void* wMat, const void* bias, const void* sequenceLengths, int batchSize,
    void* workspace, cudaStream_t stream)
{
    setupPointers(workspace);

    // Only support NZSE format for now
    // Input transpose

    doInputTranspose(x, y, init_h, init_c, final_h, final_c, sequenceLengths, batchSize, stream);


    // some calculation and setup
    int numElementsTotal = 0;
    for (int i = 0; i < maxSeqLen; i++)
    {
        numElementsTotal += static_cast<int32_t*>(mPersistentHost.validBatch.data)[i];
    }

    // Kernel call
    // cuBlas part
    T_ACCUMULATE alpha = (T_ACCUMULATE) 1.f;
    T_ACCUMULATE beta = (T_ACCUMULATE) 0.f;

    cublasSetStream(handle, stream);

    // Support multiple layers
    const char* wMatCurrent = static_cast<const char*>(wMat);
    const char* rMatCurrent = static_cast<const char*>(rMat);
    const char* biasCurrent = static_cast<const char*>(bias);

    const char* initHCurrent = static_cast<const char*>(hxT);
    const char* initCCurrent = static_cast<const char*>(cxT);

    const char* finalHCurrent = static_cast<char*>(hyT);
    const char* finalCCurrent = static_cast<char*>(cyT);

    CUASSERT(cudaMemcpyAsync(d_miniBatchArray, mPersistentHost.validBatch.data, maxSeqLen * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));

    if(param.setInitialStates)
    {
        CUASSERT(cudaMemcpyAsync((void *)finalHCurrent,(void *) initHCurrent, mScratch.hx.size,
                cudaMemcpyDeviceToDevice, stream));
    }

    for (int layerInd = 0; layerInd < param.numLayers; layerInd++)
    {
        int gemmSize = layerInd ? dirMul * param.hiddenSize : inputSize;
        const void* gemm_data = layerInd ? yT : xT;
        CUBLASASSERT(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, dirMul * param.hiddenSize * 4,
            numElementsTotal, gemmSize, &alpha, wMatCurrent, dataSize == 2 ? CUDA_R_16F : CUDA_R_32F, gemmSize,
            gemm_data, dataSize == 2 ? CUDA_R_16F : CUDA_R_32F, gemmSize, &beta, mScratch.tmpI.data,
            dataSize == 2 ? CUDA_R_16F : CUDA_R_32F, dirMul * param.hiddenSize * 4,
            sizeof(T_ACCUMULATE) == 2 ? CUDA_R_16F : CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        void* args[]
            = {(CUdeviceptr*) (&mScratch.tmpI.data), (CUdeviceptr*) (&mScratch.tmpH.data), (CUdeviceptr*) (&yT),
                (CUdeviceptr*) (&rMatCurrent), (CUdeviceptr*) (&biasCurrent), (CUdeviceptr*) (&initHCurrent),
                (CUdeviceptr*) (&initCCurrent), (CUdeviceptr*) (&finalHCurrent), (CUdeviceptr*) (&finalCCurrent),
                (CUdeviceptr*) (&d_miniBatchArray), (void*) (&effectiveSeqLen), (void*) (&numElementsTotal)};
        dim3 blockDim = dim3(param.blockSize, 1, 1);
        dim3 gridDim = dim3(param.gridSize, 1, 1);

        wrap.cuLaunchCooperativeKernel(PLSTM_tcores, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
            sharedMemoryRequired, stream, args);

        wMatCurrent += dirMul * gemmSize * param.hiddenSize * 4 * dataSize;
        rMatCurrent += dirMul * param.hiddenSize * param.hiddenSize * 4 * dataSize;
        biasCurrent += dirMul * 2 * param.hiddenSize * 4 * dataSize;

        initHCurrent += dirMul * batchSize * param.hiddenSize * dataSize;
        initCCurrent += dirMul * batchSize * param.hiddenSize * dataSize;
        finalHCurrent += dirMul * batchSize * param.hiddenSize * dataSize;
        finalCCurrent += dirMul * batchSize * param.hiddenSize * dataSize;
    }

    // Output transpose
    doOutputTranspose(y, final_h, final_c, batchSize, stream);
}

// helper function called within configurePlugin()
void CgPersistentLSTMPlugin::_createCubin()
{
    int device;
    CUASSERT(cudaGetDevice(&device));
    cudaDeviceProp deviceProp;
    CUASSERT(cudaGetDeviceProperties(&deviceProp, device));

    int dirMul = param.isBi ? 2 : 1;

    int kFragsPerWarp, mFragsPerWarp;

    int blocksPerSM = (param.gridSize + deviceProp.multiProcessorCount - 1) / deviceProp.multiProcessorCount;

    kFragsPerWarp = ((param.hiddenSize + param.blockSplitKFactor * param.warpsPerBlockK * param.FRAG_K - 1)
        / (param.blockSplitKFactor * param.warpsPerBlockK * param.FRAG_K));
    mFragsPerWarp
        = ((param.hiddenSize * 4 * param.blockSplitKFactor + param.gridSize * param.FRAG_M * param.warpsPerBlockM - 1)
            / (param.gridSize * param.FRAG_M * param.warpsPerBlockM));

    const int ldh = kFragsPerWarp * param.warpsPerBlockK * param.FRAG_K
        + ((kFragsPerWarp * param.warpsPerBlockK * param.FRAG_K) % 64 == 0 ? 8 : 0);
    size_t smemhRequired = ldh * param.innerStepSize * dataSize;
    size_t smemAccumulateRequired
        = ((param.blockSize / 32) * mFragsPerWarp * param.FRAG_M * param.innerStepSize) * sizeof(T_ACCUMULATE);

    sharedMemoryRequired = smemhRequired > smemAccumulateRequired ? smemhRequired : smemAccumulateRequired;

    if (!param.separatePath)
    {
        sharedMemoryRequired *= dirMul;
    }

    sharedMemoryRequired += dirMul * param.blockSize
        * (param.hiddenSize * maxBatchSize + param.blockSize * param.gridSize - 1) / (param.blockSize * param.gridSize)
        * dataSize;

    size_t sharedMemoryRequiredBeforeLoad = sharedMemoryRequired;
    size_t smemLoadRequired = 0;
    if (param.rfSplitFactor > 1)
    {
        size_t kFragsPerWarpInSM = kFragsPerWarp / param.rfSplitFactor;
        size_t kFragsPerBlock = param.warpsPerBlockK * param.FRAG_K * kFragsPerWarpInSM;
        size_t mFragsPerBlock = param.warpsPerBlockM * param.FRAG_M * mFragsPerWarp;
        size_t fragsPerBlock = kFragsPerBlock * mFragsPerBlock;

        if (!param.separatePath)
        {
            fragsPerBlock *= dirMul;
        }

        smemLoadRequired = fragsPerBlock * sizeof(half);

        if (param.separatePath)
        {
            size_t smemLoadRequiredtemp = param.FRAG_M * param.FRAG_K * (param.blockSize / 32) * sizeof(half);
            smemLoadRequired = smemLoadRequired > smemLoadRequiredtemp ? smemLoadRequired : smemLoadRequiredtemp;
        }

        sharedMemoryRequired = smemLoadRequired + sharedMemoryRequiredBeforeLoad;

    }
    else
    {
        smemLoadRequired = param.FRAG_M * param.FRAG_K * (param.blockSize / 32) * sizeof(half);

        if(param.separatePath)
        {
            sharedMemoryRequired = smemLoadRequired + sharedMemoryRequiredBeforeLoad;
        }
        else
        {
            sharedMemoryRequired = smemLoadRequired > sharedMemoryRequiredBeforeLoad ? smemLoadRequired : sharedMemoryRequiredBeforeLoad;
        }
    }

    int maxPerBlock;
    if (deviceProp.major == 7 && deviceProp.minor == 0) {
        // On V100 96kb is available per block with the opt-in
        maxPerBlock = 98304;
    }
    else {
        maxPerBlock = deviceProp.sharedMemPerBlock;
    }

    int maxPerSM = deviceProp.sharedMemPerMultiprocessor;

    size_t sharedMemoryAvail = std::min(maxPerBlock * blocksPerSM, maxPerSM);


    if (sharedMemoryRequired * blocksPerSM > sharedMemoryAvail)
    {
        printf("too much shared memory used. reducing rfSplitFactor...\n");
        while(sharedMemoryRequired * blocksPerSM > sharedMemoryAvail)
        {

            if (param.rfSplitFactor == 1)
            {
                printf("too big for persistent LSTM\n");
                break;
            }

            param.rfSplitFactor += 2;

            if (param.rfSplitFactor > kFragsPerWarp)
            {
                param.rfSplitFactor = 1;
            }

            if (param.rfSplitFactor > 1)
            {
                size_t kFragsPerWarpInSM = kFragsPerWarp / param.rfSplitFactor;
                size_t kFragsPerBlock = param.warpsPerBlockK * param.FRAG_K * kFragsPerWarpInSM;
                size_t mFragsPerBlock = param.warpsPerBlockM * param.FRAG_M * mFragsPerWarp;
                size_t fragsPerBlock = kFragsPerBlock * mFragsPerBlock;

                if (!param.separatePath)
                {
                    fragsPerBlock *= dirMul;
                }

                smemLoadRequired = fragsPerBlock * sizeof(half);

                if (param.separatePath)
                {
                    size_t smemLoadRequiredtemp = param.FRAG_M * param.FRAG_K * (param.blockSize / 32) * sizeof(half);
                    smemLoadRequired = std::max(smemLoadRequired, smemLoadRequiredtemp);
                }

                sharedMemoryRequired = smemLoadRequired + sharedMemoryRequiredBeforeLoad;
            }
            else
            {
                smemLoadRequired = param.FRAG_M * param.FRAG_K * (param.blockSize / 32) * sizeof(half);
                sharedMemoryRequired = std::max(smemLoadRequired, sharedMemoryRequiredBeforeLoad);
                if(param.separatePath)
                {
                    sharedMemoryRequired = smemLoadRequired + sharedMemoryRequiredBeforeLoad;
                }
                else
                {
                    sharedMemoryRequired = std::max(smemLoadRequired, sharedMemoryRequiredBeforeLoad);
                }
            }

        }
    }

    char minibatchArg[32];
    char hiddenSizeArg[32];

    char warpsPerBlockKArg[32];
    char warpsPerBlockMArg[32];

    char kFragsPerWarpArg[32];
    char mFragsPerWarpArg[32];

    char blockSplitKFactorArg[32];

    char fragMArg[32];
    char fragNArg[32];
    char fragKArg[32];

    char innerStepSizeArg[32];
    char unrollSplitKArg[32];
    char unrollGemmBatchArg[32];

    char gridDimArg[32];
    char blockDimArg[32];
    char blocksPerSMArg[32];

    char dataTypeArg[32];
    char accumulateTypeArg[32];

    char bidirectionFactorArg[32];
    char rfSplitFactorArg[32];

    snprintf(minibatchArg, 32, "-D_minibatch=%d", this->maxBatchSize);
    snprintf(hiddenSizeArg, 32, "-D_hiddenSize=%d", param.hiddenSize);

    snprintf(warpsPerBlockKArg, 32, "-D_WARPS_PER_BLOCK_K=%d", param.warpsPerBlockK);
    snprintf(warpsPerBlockMArg, 32, "-D_WARPS_PER_BLOCK_M=%d", param.warpsPerBlockM);

    snprintf(kFragsPerWarpArg, 32, "-D_kFragsPerWarp=%d", kFragsPerWarp);
    snprintf(mFragsPerWarpArg, 32, "-D_mFragsPerWarp=%d", mFragsPerWarp);

    snprintf(fragMArg, 32, "-DFRAG_M=%d", param.FRAG_M);
    snprintf(fragNArg, 32, "-DFRAG_N=%d", param.FRAG_N);
    snprintf(fragKArg, 32, "-DFRAG_K=%d", param.FRAG_K);

    snprintf(blockSplitKFactorArg, 32, "-D_blockSplitKFactor=%d", param.blockSplitKFactor);
    snprintf(innerStepSizeArg, 32, "-D_innerStepSize=%d", param.innerStepSize);
    snprintf(unrollSplitKArg, 32, "-D_unrollSplitK=%d", param.unrollSplitK);
    snprintf(unrollGemmBatchArg, 32, "-D_unrollGemmBatch=%d", param.unrollGemmBatch);

    snprintf(blockDimArg, 32, "-DBLOCK_DIM=%d", param.blockSize);
    snprintf(gridDimArg, 32, "-DGRID_DIM=%d", param.gridSize);

    snprintf(blocksPerSMArg, 32, "-DBLOCKS_PER_SM=%d", blocksPerSM);
    snprintf(dataTypeArg, 32, "-DT_DATA=%s", dataSize == 2 ? "half" : "float");
    snprintf(accumulateTypeArg, 32, "-DT_ACCUMULATE=%s", sizeof(T_ACCUMULATE) == 2 ? "half" : "float");

    snprintf(bidirectionFactorArg, 32, "-D_bidirectionFactor=%d", param.isBi ? 2 : 1);
    snprintf(rfSplitFactorArg, 32, "-DrfSplitFactor=%d", param.rfSplitFactor);

    char cudaIncludePath[1024];
    char libcudadevrtPath[1024];
    const char* cudaPath = nullptr;
    std::string token;

    // Trying to find the cuda path
    Dl_info info;
    if (dladdr((void*)cudaGetDevice, &info) != 0)
    {
        //the path should be -> /path/to/cuda/lib64/libcudart.so.10.1
        std::string s = std::string(info.dli_fname);
        std::string delimiter = "/";
        std::string removeOne = s.substr(0, s.find_last_of(delimiter));
        token = s.substr(0, removeOne.find_last_of(delimiter));
        cudaPath = token.c_str();
    }
    else
    {
        cudaPath = getenv("CUDA_PATH");
    }

    if (!cudaPath)
    {
        printf("Please set CUDA_PATH\n");
        cudaPath = "/usr/local/cuda";
    }

    int l;
    l = snprintf(cudaIncludePath, 1024, "-I%s/include", cudaPath);
    if (l > 1024)
    {
        printf("Cuda path too long.\n");
    }
    l = snprintf(libcudadevrtPath, 1024, "%s/lib64/libcudadevrt.a", cudaPath);
    if (l > 1024)
    {
        printf("Cuda path too long.\n");
    }

    std::ostringstream ss;
    int compute_num = deviceProp.major * 10 + deviceProp.minor;
    ASSERT(compute_num >= 70);
    std::string arch_str = std::string("-arch=compute_70");

    const char* tCoreCompileOpts[] = {kFragsPerWarpArg, mFragsPerWarpArg, warpsPerBlockKArg, warpsPerBlockMArg,
        minibatchArg, hiddenSizeArg, blockSplitKFactorArg, fragMArg, fragNArg, fragKArg, innerStepSizeArg,
        unrollSplitKArg, unrollGemmBatchArg, blockDimArg, gridDimArg, blocksPerSMArg, dataTypeArg, accumulateTypeArg,
        bidirectionFactorArg, rfSplitFactorArg, FTZ ? "--ftz=true" : "--ftz=false", "--use_fast_math", "-I.", cudaIncludePath,
        arch_str.c_str(), "-rdc=true"};

    CUlinkState linker;
    nvrtcProgram prog;
    cuErrCheck(wrap.cuLinkCreate(0, nullptr, nullptr, &linker), wrap);

    // Needed for co-operative groups
    cuErrCheck(wrap.cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY, libcudadevrtPath, 0, 0, 0), wrap);

    // compile
    const char* loweredNameTemp;
    if (param.separatePath)
    {
        nvrtcCompile(linker, &prog, tCoreSourceSeparate, "cgPRNN_tc.cu", "PLSTM_tcores", &loweredNameTemp, 26, tCoreCompileOpts, wrap);
    }
    else
    {
        nvrtcCompile(linker, &prog, tCoreSource, "cgPRNN_tc.cu", "PLSTM_tcores", &loweredNameTemp, 26, tCoreCompileOpts, wrap);
    }

    const std::string s(loweredNameTemp);

    size_t loweredNameSize = s.length() + 1;

    loweredName.assignNewSpace(loweredNameSize);
    strcpy(loweredName.data, s.c_str());

    void* cubinOutTemp;
    size_t cuSizeOut;
    cuErrCheck(wrap.cuLinkComplete(linker, &cubinOutTemp, &cuSizeOut), wrap);

    // Copy cubin to be saved within engine
    cubinOut.assignNewSpace(cuSizeOut);
    memcpy(cubinOut.data, cubinOutTemp, cuSizeOut);

    cuErrCheck(wrap.cuLinkDestroy(linker), wrap);

    nvrtcErrCheck(nvrtcDestroyProgram(&prog));
}

int CgPersistentLSTMPlugin::initialize()
{
    lstmRunner = new CgPersistentLSTM(this->maxBatchSize, seqLength, inputSize, dataType, param);

    assert(cubinOut.data != nullptr);
    cuErrCheck(wrap.cuModuleLoadData(&module, cubinOut.data), wrap);

    cuErrCheck(wrap.cuModuleGetFunction(&(lstmRunner->PLSTM_tcores), module, static_cast<char*>(loweredName.data)), wrap);
    cuErrCheck(wrap.cuFuncSetAttribute(
        lstmRunner->PLSTM_tcores, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, sharedMemoryRequired), wrap);

    lstmRunner->sharedMemoryRequired = sharedMemoryRequired;
    CUASSERT(cudaMalloc((void**) &(lstmRunner->d_miniBatchArray), seqLength * sizeof(int32_t)));
    return 0;
}

void CgPersistentLSTMPlugin::terminate()
{
    cuErrCheck(wrap.cuModuleUnload(module), wrap);
    if (lstmRunner->d_miniBatchArray != nullptr)
    {
        CUASSERT(cudaFree(lstmRunner->d_miniBatchArray));
    }
    lstmRunner->d_miniBatchArray = nullptr;
    delete lstmRunner;
    lstmRunner = nullptr;
}

void CgPersistentLSTMPlugin::destroy() { delete this; }


CgPersistentLSTM::~CgPersistentLSTM()
{
    if (hostScratch != nullptr)
    {
        delete[] hostScratch;
    }

    CUBLASERRORMSG(cublasDestroy(handle));
}

#endif // __x86_64__
#endif //__linux__
