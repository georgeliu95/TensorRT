#ifndef CUDA_DRIVER_WRAPPER_H
#define CUDA_DRIVER_WRAPPER_H

#ifdef __linux__
#ifdef __x86_64__
#include <cstdio>
#include <cuda.h>
#include <dlfcn.h>

namespace nvinfer1
{
class CUDADriverWrapper
{
public:
    CUDADriverWrapper();

    ~CUDADriverWrapper();

    CUresult cuGetErrorName(CUresult error, const char** pStr) const;

    CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int  value) const;

    CUresult cuLinkComplete(CUlinkState state, void** cubinOut, size_t* sizeOut) const;

    CUresult cuModuleUnload(CUmodule hmod) const;

    CUresult cuLinkDestroy(CUlinkState state) const;

    CUresult cuModuleLoadData(CUmodule* module, const void* image) const;

    CUresult cuLinkCreate(unsigned int numOptions, CUjit_option* options, void** optionValues, CUlinkState* stateOut) const;

    CUresult cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char* name) const;

    CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type, const char* path, unsigned int numOptions, CUjit_option* options, void** optionValues) const;

    CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void* data, size_t size, const char* name, unsigned int numOptions, CUjit_option* options, void** optionValues) const;

    CUresult cuLaunchCooperativeKernel (CUfunction f, unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ,
        unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ, unsigned int  sharedMemBytes, CUstream hStream, void** kernelParams) const;

private:
    void* handle;
    CUresult (*_cuGetErrorName) (CUresult, const char**);
    CUresult (*_cuFuncSetAttribute) (CUfunction, CUfunction_attribute, int);
    CUresult (*_cuLinkComplete)(CUlinkState, void**, size_t*);
    CUresult (*_cuModuleUnload)(CUmodule);
    CUresult (*_cuLinkDestroy)(CUlinkState);
    CUresult (*_cuLinkCreate)(unsigned int, CUjit_option*, void**, CUlinkState*);
    CUresult (*_cuModuleLoadData)(CUmodule*, const void*);
    CUresult (*_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*_cuLinkAddFile)(CUlinkState, CUjitInputType, const char*, unsigned int, CUjit_option*, void**);
    CUresult (*_cuLinkAddData)(
        CUlinkState, CUjitInputType, void*, size_t, const char*, unsigned int, CUjit_option*, void**);
    CUresult (*_cuLaunchCooperativeKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
        unsigned int, unsigned int, unsigned int, CUstream, void**);
};
} // namespace nvinfer1
#endif // __x86_64__
#endif //__linux__
#endif // CUDA_DRIVER_WRAPPER_H
