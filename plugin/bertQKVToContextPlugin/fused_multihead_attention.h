/***************************************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once
#include "cudaDriverWrapper.h"
#include "cuda_runtime_api.h"
#include <mutex>
#include <assert.h>
#include <stdint.h>
#include <unordered_map>
#include <vector>

namespace bert
{

static constexpr int32_t kSM_TURING = 75;
static constexpr int32_t kSM_AMPERE = 80;

enum Data_type
{
    DATA_TYPE_BOOL,
    DATA_TYPE_E8M10,
    DATA_TYPE_E8M7,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32
};

static inline size_t get_size_in_bytes(size_t n, Data_type dtype)
{
    switch (dtype)
    {
    case DATA_TYPE_E8M10: return n * 4;
    case DATA_TYPE_FP32: return n * 4;
    case DATA_TYPE_FP16: return n * 2;
    case DATA_TYPE_INT32: return n * 4;
    case DATA_TYPE_INT8: return n;
    case DATA_TYPE_INT4: return n / 2;
    case DATA_TYPE_BOOL: return n / 8;
    case DATA_TYPE_E8M7: return n * 2;
    default: assert(false); return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Fused_multihead_attention_params
{
    // The QKV matrices.
    void* qkv_ptr;
    // The mask to implement drop-out.
    void* packed_mask_ptr;
    // The O matrix (output).
    void* o_ptr;

    // The stride between rows of the Q, K and V matrices.
    int64_t qkv_stride_in_bytes;
    // The stride between matrices of packed mask.
    int64_t packed_mask_stride_in_bytes;
    // The stride between rows of O.
    int64_t o_stride_in_bytes;

#if defined(STORE_P)
    // The pointer to the P matrix (for debugging).
    void* p_ptr;
    // The stride between rows of the P matrix (for debugging).
    int64_t p_stride_in_bytes;
#endif // defined(STORE_P)

#if defined(STORE_S)
    // The pointer to the S matrix (for debugging).
    void* s_ptr;
    // The stride between rows of the S matrix (for debugging).
    int64_t s_stride_in_bytes;
#endif // defined(STORE_S)

    // The dimensions.
    int b, h, s, d;
    // The scaling factors for the kernel.
    uint32_t scale_bmm1, scale_softmax, scale_bmm2;

    // Do we use Niall's trick to avoid I2F/F2I in the INT8 kernel.
    // See https://confluence.nvidia.com/pages/viewpage.action?pageId=302779721 for details.
    bool enable_i2f_trick;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
extern unsigned char fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_128_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_384_64_kernel_sm75_cu_o[];
extern unsigned char fused_multihead_attention_int8_384_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_int8_128_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o[];
extern unsigned char fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o[];

extern unsigned int fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_int8_128_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_int8_384_64_kernel_sm75_cu_o_len;
extern unsigned int fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len;
extern unsigned int fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o_len;

static const struct FusedMultiHeadAttentionKernelMetaInfo
{
    Data_type mDataType;
    unsigned int mS;
    unsigned int mD;
    unsigned int mSM;
    const unsigned char* mCubin;
    unsigned int mCubinSize;
    const char* mFuncName;
    unsigned int mSharedMemBytes;
    unsigned int mThreadsPerCTA;
} sMhaKernelMetaInfos[] = {
    // Turing
    {DATA_TYPE_FP16, 128, 64, kSM_TURING, fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm75_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm75",
        32768, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_TURING, fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o,
        fused_multihead_attention_fp16_384_64_kernel_sm75_cu_o_len, "fused_multihead_attention_fp16_384_64_kernel_sm75",
        57344, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_TURING, fused_multihead_attention_int8_128_64_kernel_sm75_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm75_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm75",
        16384, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_TURING, fused_multihead_attention_int8_384_64_kernel_sm75_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm75_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm75",
        53284, 256},
#if CUDA_VERSION >= 11000
    // Ampere
    {DATA_TYPE_FP16, 128, 64, kSM_AMPERE, fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_128_64_kernel_sm80",
        49152, 128},
    {DATA_TYPE_FP16, 384, 64, kSM_AMPERE, fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_fp16_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_fp16_384_64_kernel_sm80",
        114688, 256},
    {DATA_TYPE_INT8, 128, 64, kSM_AMPERE, fused_multihead_attention_int8_128_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_128_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_128_64_kernel_sm80",
        24576, 128},
    {DATA_TYPE_INT8, 384, 64, kSM_AMPERE, fused_multihead_attention_int8_384_64_kernel_sm80_cu_o,
        fused_multihead_attention_int8_384_64_kernel_sm80_cu_o_len, "fused_multihead_attention_int8_384_64_kernel_sm80",
        57344, 256},
#endif
};

struct FusedMultiHeadAttentionXMMAKernel
{
    inline uint64_t hashID(unsigned int s, unsigned int d) const
    {
        return (uint64_t) s << 32 | d;
    }

    FusedMultiHeadAttentionXMMAKernel(Data_type type, unsigned int sm)
        : mDataType(type)
        , mSM(sm)
    {
        for (unsigned int i = 0; i < sizeof(sMhaKernelMetaInfos) / sizeof(sMhaKernelMetaInfos[0]); ++i)
        {
            const auto& kernelMeta = sMhaKernelMetaInfos[i];
            if (kernelMeta.mSM == sm && kernelMeta.mDataType == type)
            {
                CUmodule hmod{0};
                cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                mModules.push_back(hmod);

                FusedMultiHeadAttentionKernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= 48 * 1024)
                {
                    cuErrCheck(mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                   CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes),
                        mDriver);
                }
                mFunctions.insert(std::make_pair(hashID(kernelMeta.mS, kernelMeta.mD), funcInfo));
            }
        }
    }

    ~FusedMultiHeadAttentionXMMAKernel()
    {
        for (auto mod : mModules)
        {
            mDriver.cuModuleUnload(mod);
        }
        mFunctions.clear();
        mModules.clear();
    }

    bool isValid() const
    {
        return !mFunctions.empty();
    }

    void run(Fused_multihead_attention_params& params, size_t s, size_t d, cudaStream_t ss) const
    {
        const auto findIter = mFunctions.find(hashID(s, d));
        if (findIter != mFunctions.end())
        {
            const auto& kernelMeta = sMhaKernelMetaInfos[findIter->second.mMetaInfoIndex];
            const CUfunction func = findIter->second.mDeviceFunction;

            void* kernelParams[] = {&params, nullptr};
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
        else
        {
            ASSERT(0);
        }
    }

    nvinfer1::CUDADriverWrapper mDriver;

    Data_type mDataType;
    unsigned int mSM;
    std::vector<CUmodule> mModules;
    struct FusedMultiHeadAttentionKernelInfo
    {
        unsigned int mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };
    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
};


class FusedMultiHeadAttentionXMMAKernelFactory
{
public:
    ~FusedMultiHeadAttentionXMMAKernelFactory()
    {
        for (auto kernelPair : mKernels)
        {
            delete kernelPair.second;
        }
        mKernels.clear();
    }

    const FusedMultiHeadAttentionXMMAKernel* getXMMAKernels(Data_type type, unsigned int sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        const auto id = hashID(type, sm);
        const auto findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            const FusedMultiHeadAttentionXMMAKernel* newKernel = new FusedMultiHeadAttentionXMMAKernel{type, sm};
            mKernels.insert(std::make_pair(id, newKernel));
            return newKernel;
        }
        return findIter->second;
    }

    static FusedMultiHeadAttentionXMMAKernelFactory& Get()
    {
        static FusedMultiHeadAttentionXMMAKernelFactory s_factory;
        return s_factory;
    }

private:
    FusedMultiHeadAttentionXMMAKernelFactory(){}

    inline uint64_t hashID(Data_type type, unsigned int sm) const
    {
        return (uint64_t) type << 32 | sm;
    }

    std::unordered_map<uint64_t, const FusedMultiHeadAttentionXMMAKernel*> mKernels;
};

} // namespace bert
