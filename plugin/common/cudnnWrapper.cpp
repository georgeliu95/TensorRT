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

#include "cudnnWrapper.h"
#include "common/checkMacrosPlugin.h"

#define CUDNN_MAJOR 8
#if defined(_WIN32)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif // defined(WIN32_LEAN_AND_MEAN)
#include <windows.h>
#define dllOpen(name) (void*) LoadLibraryA(name)
#define dllClose(handle) FreeLibrary(static_cast<HMODULE>(handle))
#define dllGetSym(handle, name) GetProcAddress(static_cast<HMODULE>(handle), name)
auto const kCUDNN_PLUGIN_LIBNAME = std::string("cudnn64_") + std::to_string(CUDNN_MAJOR) + ".dll";
#else
#include <dlfcn.h>
#define dllOpen(name) dlopen(name, RTLD_LAZY)
#define dllClose(handle) dlclose(handle)
#define dllGetSym(handle, name) dlsym(handle, name)
auto const kCUDNN_PLUGIN_LIBNAME = std::string("libcudnn.so.") + std::to_string(CUDNN_MAJOR);
#endif

namespace nvinfer1
{
namespace pluginInternal
{
using namespace nvinfer1;

CudnnWrapper::CudnnWrapper()
{
    mLibrary = dllOpen(kCUDNN_PLUGIN_LIBNAME.c_str());
    mFail2LoadMsg = "Failed to load " + kCUDNN_PLUGIN_LIBNAME + ".";
    // Cudnn library is available
    if (mLibrary != nullptr)
    {
        PLUGIN_ASSERT(mLibrary != nullptr);
        _cudnnCreate = reinterpret_cast<cudnnCreateType>(dllGetSym(mLibrary, "cudnnCreate"));
        _cudnnDestroy = reinterpret_cast<cudnnDestroyType>(dllGetSym(mLibrary, "cudnnDestroy"));
        PLUGIN_VALIDATE_MSG(cudnnCreate(&mHandle) == CUDNN_STATUS_SUCCESS, "Could not create cudnn handle.");
        PLUGIN_ASSERT(mHandle != nullptr);
    }
    else
    {
        mHandle = nullptr;
        nvinfer1::plugin::gLogWarning << "Unable to dynamically load " << kCUDNN_PLUGIN_LIBNAME
                                      << ". Cudnn handle is set to nullptr. Please provide the library if needed."
                                      << std::endl;
    }
}

CudnnWrapper::~CudnnWrapper()
{
    if (mHandle != nullptr)
    {
        PLUGIN_VALIDATE_MSG(cudnnDestroy(mHandle) == CUDNN_STATUS_SUCCESS, "Could not destroy cudnn handle.");
        mHandle = nullptr;
    }

    if (mLibrary != nullptr)
    {
        dllClose(mLibrary);
    }
}

cudnnContext* CudnnWrapper::getCudnnHandle() const
{
    return mHandle;
}

bool CudnnWrapper::isValid() const
{
    return mHandle != nullptr;
}

cudnnStatus_t CudnnWrapper::cudnnCreate(cudnnContext** handle)
{
    PLUGIN_VALIDATE_MSG(mLibrary != nullptr, mFail2LoadMsg.c_str());
    return _cudnnCreate(handle);
}

cudnnStatus_t CudnnWrapper::cudnnDestroy(cudnnContext* handle)
{
    PLUGIN_VALIDATE_MSG(mLibrary != nullptr, mFail2LoadMsg.c_str());
    return _cudnnDestroy(handle);
}

} // namespace pluginInternal
} // namespace nvinfer1
