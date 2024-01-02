/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef TRT_PLUGIN_CUDNN_WRAPPER_H
#define TRT_PLUGIN_CUDNN_WRAPPER_H

#include "NvInferPlugin.h"
#include <string>

namespace nvinfer1
{
namespace pluginInternal
{
/*
 * Copy of the CUDNN return codes
 */
enum CudnnStatus
{
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13,
    CUDNN_STATUS_VERSION_MISMATCH = 14,
};
using cudnnStatus_t = CudnnStatus;
using cudnnHandle_t = struct cudnnContext*;

using cudnnCreateType = cudnnStatus_t (*)(cudnnContext**);
using cudnnDestroyType = cudnnStatus_t (*)(cudnnContext*);

class CudnnWrapper
{
public:
    CudnnWrapper();
    ~CudnnWrapper();

    cudnnContext* getCudnnHandle() const;
    bool isValid() const;

    cudnnStatus_t cudnnCreate(cudnnContext** handle);
    cudnnStatus_t cudnnDestroy(cudnnContext* handle);

private:
    void* mLibrary{nullptr};
    cudnnContext* mHandle{nullptr};
    std::string mFail2LoadMsg;
    cudnnCreateType _cudnnCreate{nullptr};
    cudnnDestroyType _cudnnDestroy{nullptr};
};
} // namespace pluginInternal
} // namespace nvinfer1

#endif // TRT_PLUGIN_CUDNN_WRAPPER_H
