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

#include "cublasWrapper.h"
#include "common/checkMacrosPlugin.h"
#include "cudaDriverWrapper.h"

#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void*) LoadLibraryA(name)
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) GetProcAddress(static_cast<HMODULE>(handle), name)
auto const kCUBLAS_PLUGIN_LIBNAME
    = std::string{"cublas64_"} + std::to_string(nvinfer1::getCudaLibVersionMaj()) + ".dll";
#else
#include <dlfcn.h>
#define dllOpen(name) dlopen(name, RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
auto const kCUBLAS_PLUGIN_LIBNAME = std::string{"libcublas.so."} + std::to_string(nvinfer1::getCudaLibVersionMaj());
#endif

namespace nvinfer1
{
namespace pluginInternal
{
using namespace nvinfer1;

CublasWrapper::CublasWrapper()
{
    mLibrary = dllOpen(kCUBLAS_PLUGIN_LIBNAME.c_str());
    mFail2LoadMsg = "Failed to load " + kCUBLAS_PLUGIN_LIBNAME + ".";
    // cublas library is available
    if (mLibrary != nullptr)
    {
        PLUGIN_ASSERT(mLibrary != nullptr);
        _cublasCreate = reinterpret_cast<cublasCreateType>(dllGetSym(mLibrary, "cublasCreate_v2"));
        _cublasDestroy = reinterpret_cast<cublasDestroyType>(dllGetSym(mLibrary, "cublasDestroy_v2"));
        PLUGIN_VALIDATE_MSG(cublasCreate(&mHandle) == CUBLAS_STATUS_SUCCESS, "Could not create cublas handle.");
        PLUGIN_ASSERT(mHandle != nullptr);
    }
    else
    {
        mHandle = nullptr;
        nvinfer1::plugin::gLogWarning << "Unable to dynamically load " << kCUBLAS_PLUGIN_LIBNAME
                                      << ". Cublas handle is set to nullptr. Please provide the library if needed."
                                      << std::endl;
    }
}

CublasWrapper::~CublasWrapper()
{
    if (mHandle != nullptr)
    {
        PLUGIN_VALIDATE_MSG(cublasDestroy(mHandle) == CUBLAS_STATUS_SUCCESS, "Could not destroy cublas handle.");
        mHandle = nullptr;
    }

    if (mLibrary != nullptr)
    {
        dllClose(mLibrary);
    }
}

cublasContext* CublasWrapper::getCublasHandle() const
{
    return mHandle;
}

bool CublasWrapper::isValid() const
{
    return mHandle != nullptr;
}

cublasStatus_t CublasWrapper::cublasCreate(cublasContext** handle)
{
    PLUGIN_VALIDATE_MSG(mLibrary != nullptr, mFail2LoadMsg.c_str());
    return _cublasCreate(handle);
}

cublasStatus_t CublasWrapper::cublasDestroy(cublasContext* handle)
{
    PLUGIN_VALIDATE_MSG(mLibrary != nullptr, mFail2LoadMsg.c_str());
    return _cublasDestroy(handle);
}

} // namespace pluginInternal
} // namespace nvinfer1
