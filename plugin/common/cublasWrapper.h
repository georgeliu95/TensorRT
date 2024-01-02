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

#ifndef TRT_PLUGIN_CUBLAS_WRAPPER_H
#define TRT_PLUGIN_CUBLAS_WRAPPER_H

#include "NvInferPlugin.h"
#include <string>

namespace nvinfer1
{
namespace pluginInternal
{
/* Copy of CUBLAS status type returns */
enum CublasStatus
{
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
};
using cublasStatus_t = CublasStatus;
using cublasHandle_t = struct cublasContext*;

using cublasCreateType = cublasStatus_t (*)(cublasContext**);
using cublasDestroyType = cublasStatus_t (*)(cublasContext*);

class CublasWrapper
{
public:
    CublasWrapper();
    ~CublasWrapper();

    cublasContext* getCublasHandle() const;
    bool isValid() const;

    cublasStatus_t cublasCreate(cublasContext** handle);
    cublasStatus_t cublasDestroy(cublasContext* handle);

private:
    void* mLibrary{nullptr};
    cublasContext* mHandle{nullptr};
    std::string mFail2LoadMsg;
    cublasCreateType _cublasCreate{nullptr};
    cublasDestroyType _cublasDestroy{nullptr};
};
} // namespace pluginInternal
} // namespace nvinfer1

#endif // TRT_PLUGIN_CUBLAS_WRAPPER_H
