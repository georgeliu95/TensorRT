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
#ifndef CG_PERSISTENT_LSTM_H
#define CG_PERSISTENT_LSTM_H

#ifdef __linux__
#if (defined(__x86_64__) || defined(__PPC__))

#include "NvInferPlugin.h"
#include "cudaDriverWrapper.h"
#include "checkMacros.h"
#include "legacy_plugin.h"
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <nvrtc.h>
#include <string>

#define T_ACCUMULATE float
#define FTZ 0 // Flush to zero flag

// copied from cudaRNNBaseRunner.cpp
struct SizedRegion
{
    size_t size{0};
    void* data{nullptr};
};

class AllocatedSizedRegion
{
public:
    AllocatedSizedRegion()
    {
        size = 0;
        data = nullptr;
    }

    inline ~AllocatedSizedRegion()
    {
        if (data != nullptr)
        {
            free(data);
        }
        size = 0;
    }

    inline void assignNewSpace(size_t new_size)
    {
        if (data != nullptr)
        {
            free(data);
        }
        data = static_cast<char*>(malloc(new_size));
        this->size = new_size;
    }

    size_t size{0};
    char* data{nullptr};
};

struct CgPLSTMParameters
{
    int hiddenSize{0};
    int numLayers{0};

    int warpsPerBlockK{0};
    int warpsPerBlockM{0};
    int blockSplitKFactor{0};

    int FRAG_M{0};
    int FRAG_N{0};
    int FRAG_K{0};

    int innerStepSize{0};
    int unrollSplitK{0};
    int unrollGemmBatch{0}; // current disabled

    int gridSize{0};
    int blockSize{0};

    int rfSplitFactor{0};

    bool isBi{false};
    bool separatePath{false};
    bool setInitialStates{false};
};

namespace nvinfer1
{
namespace plugin
{

class CgPersistentLSTM
{
public:
    CgPersistentLSTM(int batchSize, int seqLen, int inputSize, nvinfer1::DataType dataType, CgPLSTMParameters param);

    ~CgPersistentLSTM();

    void setupScratchSizes();

    void setupPersistentSizes();

    size_t computeScratchSize() const;

    size_t computePersistentHostSize() const;

    static size_t computeGpuSize(int maxBatchSize, int maxSeqLen, int inputSize, int dataSize, CgPLSTMParameters param);

    void setupPointers(void* deviceScratch);

    void doInputTranspose(const void* x, void* y, const void* hx, const void* cx, void* hy, void* cy,
        const void* sequenceLengths, int inputBatchSize, cudaStream_t stream);

    void doOutputTranspose(void* y, void* hy, void* cy, int inputBatchSize, cudaStream_t stream);

    void setupSequenceLengths(int32_t* lengths, int inputBatchSize, const void* sequenceLengths,
        cudaStream_t stream); // assume it to be in gpu

    void execute(const void* x, void* y, const void* init_h, const void* init_c, void* final_h, void* final_c,
        const void* rMat, const void* wMat, const void* bias, const void* sequenceLengths, int batchSize,
        void* workspace, cudaStream_t stream);

    CUfunction PLSTM_tcores;
    size_t sharedMemoryRequired{0};
    int* d_miniBatchArray{nullptr};
    char* hostScratch{nullptr};

private:
    int maxBatchSize{0}, maxSeqLen{0}, effectiveSeqLen{0}, dataSize{0}, inputSize{0};
    CgPLSTMParameters param;
    nvinfer1::DataType dataType{nvinfer1::DataType::kHALF};
    int dirMul{0};

    const void *xT{nullptr}, *hxT{nullptr}, *cxT{nullptr};
    void *yT{nullptr}, *hyT{nullptr}, *cyT{nullptr}, *gpuX{nullptr}, *gpuY{nullptr};
    int32_t *cpuPermutation{nullptr}, *cpuLengths{nullptr};
    int32_t *gpuPermutation{nullptr}, *gpuLengths{nullptr};
    bool NZSE2SNZE{false}, NZLH2LNZH{false};

    struct RNNPersistent
    {
        SizedRegion permutation, lengths, x, y, validBatch;
    };

    struct RNNScratch
    {
        // Device scratch regions
        SizedRegion tmpI, tmpH, permutation, lengths, x, y, ix, oy, hx, cx, hy, cy;
    };

    RNNPersistent mPersistentHost;
    RNNScratch mScratch;
    cublasHandle_t handle;
    CUDADriverWrapper wrap;
};
} // namespace plugin
} // namespace nvinfer1

#endif // (defined(__x86_64__) || defined(__PPC__))
#endif //__linux__

#endif // CG_PERSISTENT_LSTM_H
