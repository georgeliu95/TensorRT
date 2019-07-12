#ifdef __linux__
#ifdef __x86_64__
#include <cuda.h>
#include <stdio.h>
#include <dlfcn.h>
#include "cudaDriverWrapper.h"
#include "checkMacros.h"
#include "helpers.h"

using namespace nvinfer1;

CUDADriverWrapper::CUDADriverWrapper()
{
    handle = dlopen("libcuda.so.1", RTLD_LAZY);
    ASSERT(handle != nullptr);

    auto load_sym = [](void *handle, const char *name) {
        void *ret = dlsym(handle, name);
        ASSERT(ret != nullptr);
        return ret;
    };

    *(void**)(&_cuGetErrorName) = load_sym(handle, "cuGetErrorName");
    *(void**)(&_cuFuncSetAttribute) = load_sym(handle, "cuFuncSetAttribute");
    *(void**)(&_cuLinkComplete) = load_sym(handle, "cuLinkComplete");
    *(void**)(&_cuModuleUnload) = load_sym(handle, "cuModuleUnload");
    *(void**)(&_cuLinkDestroy) = load_sym(handle, "cuLinkDestroy");
    *(void**)(&_cuModuleLoadData) = load_sym(handle, "cuModuleLoadData");
    *(void**)(&_cuLinkCreate) = load_sym(handle, "cuLinkCreate_v2");
    *(void**)(&_cuModuleGetFunction) = load_sym(handle, "cuModuleGetFunction");
    *(void**)(&_cuLinkAddFile) = load_sym(handle, "cuLinkAddFile_v2");
    *(void**)(&_cuLinkAddData) = load_sym(handle, "cuLinkAddData_v2");
    *(void**)(&_cuLaunchCooperativeKernel) = load_sym(handle, "cuLaunchCooperativeKernel");
}

CUDADriverWrapper::~CUDADriverWrapper()
{
    dlclose(handle);
}

CUresult CUDADriverWrapper::cuGetErrorName(CUresult error, const char** pStr) const
{
    return (*_cuGetErrorName)(error, pStr);
}

CUresult CUDADriverWrapper::cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int  value) const
{
    return (*_cuFuncSetAttribute)(hfunc, attrib, value);
}

CUresult CUDADriverWrapper::cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const
{
    return (*_cuLinkComplete)(state, cubinOut, sizeOut);
}

CUresult CUDADriverWrapper::cuModuleUnload(CUmodule hmod) const
{
    return (*_cuModuleUnload)(hmod);
}

CUresult CUDADriverWrapper::cuLinkDestroy(CUlinkState state) const
{
    return (*_cuLinkDestroy)(state);
}

CUresult CUDADriverWrapper::cuModuleLoadData(CUmodule* module, const void* image) const
{
    return (*_cuModuleLoadData)(module, image);
}

CUresult CUDADriverWrapper::cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const
{
    return (*_cuLinkCreate)(numOptions, options, optionValues, stateOut);
}

CUresult CUDADriverWrapper::cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) const
{
    return (*_cuModuleGetFunction)(hfunc, hmod, name);
}

CUresult CUDADriverWrapper::cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddFile)(state, type, path, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) const
{
    return (*_cuLinkAddData)(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult CUDADriverWrapper::cuLaunchCooperativeKernel (CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ,
    unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams) const
{
    return (*_cuLaunchCooperativeKernel)(f, gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams);
}

#endif // __x86_64__
#endif //__linux__
