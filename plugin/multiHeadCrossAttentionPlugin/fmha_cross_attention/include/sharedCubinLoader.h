/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#ifndef _SHARED_CUBIN_LOADER_
#define _SHARED_CUBIN_LOADER_
#include "common/cudaDriverWrapper.h"
#include "commonDatatype.h"
#include "cuda_runtime_api.h"
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
template <typename TKernelMeta, typename TKernelParam>
class TSharedCubinKernel
{
public:
    using KernelMeta = TKernelMeta;
    using KernelParam = TKernelParam;

    virtual uint64_t hashID(const KernelMeta& kernelMeta) const = 0;
    virtual uint64_t hashID(const TKernelParam& param) const = 0;

    TSharedCubinKernel(const TKernelMeta* pMetaStart, uint32_t nMetaCount, Data_type type, uint32_t sm)
        : mDataType(type)
        , mKernelMeta(pMetaStart)
        , mKernelMetaCount(nMetaCount)
        , mSM(sm)
    {
        PLUGIN_ASSERT(mKernelMetaCount && "No kernels were loaded correctly.");
    }

    void loadCubinKernels(uint32_t smVersion)
    {
        for (uint32_t i = 0; i < mKernelMetaCount; ++i)
        {
            const auto& kernelMeta = mKernelMeta[i];
            const auto kernelKey = hashID(kernelMeta);
            if (kernelMeta.mSM == smVersion && kernelMeta.mDataType == mDataType
                && mFunctions.find(kernelKey) == mFunctions.end())
            {
                const uint32_t DEFAULT_SMEM_SIZE{48 * 1024};
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE)
                {
                    int32_t deviceID{0};
                    cudaGetDevice(&deviceID);
                    int32_t sharedMemPerMultiprocessor{0};
                    if (cudaDeviceGetAttribute(
                            &sharedMemPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID)
                            != cudaSuccess
                        || sharedMemPerMultiprocessor < static_cast<int32_t>(kernelMeta.mSharedMemBytes))
                    {
                        // skip load function because not enough shared memory to launch the kernel
                        continue;
                    }
                }

                CUmodule hmod{0};
                auto findModuleIter = mModules.find(kernelMeta.mCubin);
                if (findModuleIter != mModules.end())
                {
                    hmod = findModuleIter->second;
                }
                else
                {
                    cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                    mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
                }

                FusedMultiHeadAttentionKernelInfo funcInfo;
                funcInfo.mMetaInfoIndex = i;
                cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
                if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE)
                {
                    if (mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernelMeta.mSharedMemBytes)
                        != CUDA_SUCCESS)
                    {
                        // some chip may not have enough shared memory to launch the kernel
                        continue;
                    }
                }
                mFunctions.insert({kernelKey, funcInfo});
                const int s = static_cast<int>(kernelMeta.mS);
                if (mValidSequences.find(s) == mValidSequences.end())
                {
                    mValidSequences.insert(s);
                }
            }
        }
    }

    void loadCubinKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }

        loadCubinKernels(mSM);

        // sm_86 chips prefer sm_86 sass, but can also use sm_80 sass if sm_86 not exist.
        // sm_87 cannot run sm_80 sass
        if (mSM == kSM_86)
        {
            loadCubinKernels(kSM_80);
        }
    }

    bool isValid(int s) const
    {
        return (mValidSequences.find(s) != mValidSequences.end());
    }

    virtual void run(TKernelParam& params, cudaStream_t ss) const
    {
        if (params.interleaved)
        {
            PLUGIN_ASSERT(mDataType == DATA_TYPE_INT8);
        }

        const auto findIter = mFunctions.find(hashID(params));
        // Provide debug information if the kernel is missing in the pool.
        std::stringstream configss;
        configss << "s: " << params.s << " d: " << params.d << " interleaved?: " << params.interleaved
                 << " forceUnroll?: " << params.force_unroll;
        PLUGIN_ASSERT(findIter != mFunctions.end() && configss.str().c_str());

        const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;

        void* kernelParams[] = {&params, nullptr};
        if (!params.force_unroll)
        {
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
        else
        {
            int32_t unroll = kernelMeta.mS / kernelMeta.mUnrollStep;
            PLUGIN_ASSERT(kernelMeta.mS == kernelMeta.mUnrollStep * unroll);
            cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, unroll, kernelMeta.mThreadsPerCTA, 1, 1,
                           kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                mDriver);
        }
    }

    virtual ~TSharedCubinKernel() = default;

protected:
    nvinfer1::CUDADriverWrapper mDriver;

    Data_type mDataType;
    const TKernelMeta* mKernelMeta;
    uint32_t mKernelMetaCount;
    uint32_t mSM;
    std::unordered_map<const unsigned char*, CUmodule> mModules;
    struct FusedMultiHeadAttentionKernelInfo
    {
        uint32_t mMetaInfoIndex;
        CUfunction mDeviceFunction;
    };
    std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
    std::set<int> mValidSequences;
};

template <typename TKernelList>
class TSharedCubinKernelFactory
{
public:
    const TKernelList* getCubinKernels(
        const typename TKernelList::KernelMeta* pKernelList, uint32_t nbKernels, Data_type type, uint32_t sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        const auto id = hashID(type, sm);
        const auto findIter = mKernels.find(id);
        if (findIter == mKernels.end())
        {
            TKernelList* newKernel = new TKernelList{pKernelList, nbKernels, type, sm};
            newKernel->loadCubinKernels();
            mKernels.insert(std::make_pair(id, std::unique_ptr<TKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static TSharedCubinKernelFactory<TKernelList>& Get()
    {
        static TSharedCubinKernelFactory<TKernelList> s_factory;
        return s_factory;
    }

private:
    TSharedCubinKernelFactory() = default;

    inline uint64_t hashID(Data_type type, uint32_t sm) const
    {
        // use deviceID in hasID for multi GPU support before driver support context-less loading of cubin
        int32_t deviceID{0};
        CSC(cudaGetDevice(&deviceID), STATUS_FAILURE);

        PLUGIN_ASSERT((deviceID & 0xFFFF) == deviceID);
        PLUGIN_ASSERT((type & 0xFFFF) == type);
        PLUGIN_ASSERT((sm & 0xFFFFFFFF) == sm);
        return (uint64_t) type << 48 | (uint64_t) deviceID << 32 | sm;
    }

    std::unordered_map<uint64_t, const std::unique_ptr<TKernelList>> mKernels;
};

} // namespace plugin
} // namespace nvinfer1
#endif // _SHARED_CUBIN_LOADER_
