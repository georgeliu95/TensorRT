/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <assert.h>
#include "bind_data.h"
#include "error_util.h"
#include "type_utils.h"
#include "cublas_v2.h" // for __half2
#include "logger.h"
#include "common.h"

namespace RNNDataUtil
{

template <typename value_type>
BindData<value_type>::BindData(const nvinfer1::ICudaEngine* engine, const std::vector<std::string>& names,
                               std::vector<uint32_t> sizes,
                               uint32_t numOutputs,
                               int layerID)
    : mNumBindings(names.size())
    , mNumOutputs(numOutputs)
    , mNames(names)
    , mSizes(sizes)
    , mEnginePtr(engine)

{
    mIndices = new int[mNumBindings];
    mBuffers = new void*[mNumBindings];
    mData = new value_type*[mNumBindings];
    std::fill(mBuffers, mBuffers + mNumBindings, nullptr);
    std::fill(mData, mData + mNumBindings, nullptr);
    std::fill(mIndices, mIndices + mNumBindings, -1);
    assert(mEnginePtr->getNbBindings() == (int) mNumBindings);
    if (layerID >= 0)
    {
        for (auto& str : mNames)
        {
            std::stringstream ss;
            ss << str << layerID;
            str = ss.str();
        }
    }
    allocate();
}

template <typename value_type>
void BindData<value_type>::copyDataToDevice(std::vector<uint32_t> inputIds,
                                            const cudaStream_t& stream)
{
    for (auto z : inputIds)
    {
        CHECK(cudaMemcpyAsync(mBuffers[mIndices[z]], mData[z], mSizes[z] * sizeof(value_type),
                              cudaMemcpyHostToDevice, stream));
    }
}

template <typename value_type>
void BindData<value_type>::copyDataToDevice(std::vector<uint32_t> inputIds,
                                            std::vector<uint32_t>& hostOffsets,
                                            std::vector<uint32_t>& hostSizes,
                                            const cudaStream_t& stream)
{
    for (auto z : inputIds)
    {
        CHECK(cudaMemcpyAsync(mBuffers[mIndices[z]], mData[z] + hostOffsets[z], hostSizes[z] * sizeof(value_type),
                              cudaMemcpyHostToDevice, stream));
    }
}

template <typename value_type>
void BindData<value_type>::copyDataFromDevice(std::vector<uint32_t> outputIds, const cudaStream_t& stream)
{
    for (auto z : outputIds)
        CHECK(cudaMemcpyAsync(mData[z], mBuffers[mIndices[z]], mSizes[z] * sizeof(value_type), cudaMemcpyDeviceToHost, stream));
}

template <typename value_type>
bool BindData<value_type>::allocate() // takes engine
{
    std::fill(mIndices, mIndices + mNumBindings, -1);
    for (uint32_t x = 0; x < mNumBindings; ++x)
    {
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // note that indices are guaranteed to be less than IEngine::getNbBindings()
        mIndices[x] = mEnginePtr->getBindingIndex(mNames[x].c_str());
        if (mIndices[x] == -1)
            continue;
        // create GPU buffers and a stream
        assert(mIndices[x] < (int) mNumBindings);
        if (std::is_same<value_type, half1>::value)
        {
            // make sure data is fit to support half2
            if (mSizes[x] % 2 != 0)
                FatalError("Expect even buffer's size in kHALF2 mode");
            CHECK(cudaMalloc(&mBuffers[mIndices[x]], mSizes[x] / 2 * sizeof(half2)));
        }
        else
        {
            CHECK(cudaMalloc(&mBuffers[mIndices[x]], mSizes[x] * sizeof(value_type)));
        }
        mData[x] = new value_type[mSizes[x]];
    }
    // Initialize input/hidden/cell state to zero
    Convert<value_type> fromFloat;
    for (uint32_t x = 0; x < mNumBindings; ++x)
        std::fill(mData[x], mData[x] + mSizes[x], fromFloat(0.0f));
    return 0;
}

template <typename value_type>
bool BindData<value_type>::deallocate()
{
    for (uint32_t x = 0; x < mNumBindings; ++x)
    {
        if (mBuffers[mIndices[x]])
            CHECK(cudaFree(mBuffers[mIndices[x]]));
        if (mData[x])
            delete[] mData[x];
    }
    return 0;
}

template <typename value_type>
BindData<value_type>::~BindData()
{
    deallocate();
    if (mIndices != nullptr)
        delete[] mIndices;
    if (mBuffers != nullptr)
        delete[] mBuffers;
    if (mData != nullptr)
        delete[] mData;
}

template class BindData<float>;
template class BindData<half1>;
} // end namespace RNNDataUtil
