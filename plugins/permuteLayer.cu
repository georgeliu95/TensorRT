#include "NvInferPlugin.h"
#include "ssd.h"
#include "ssdMacros.h"

using nvinfer1::plugin::Quadruple;

namespace nvinfer1
{
namespace plugin
{

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void permuteKernel(
        const int n,
        const Quadruple permuteOrder,
        const Quadruple oldSteps,
        const Quadruple newSteps,
        const float* const inputData,
        float* const outputData)
{
    for (int index = blockIdx.x * nthdsPerCTA + threadIdx.x;
         index < n; index += gridDim.x * nthdsPerCTA)
    {
        int tempIdx = index;
        int oldIdx = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            int order = permuteOrder.data[i];
            oldIdx += (tempIdx / newSteps.data[i]) * oldSteps.data[order];
            tempIdx %= newSteps.data[i];
        }
        outputData[index] = inputData[oldIdx];
    }
}

ssdStatus_t permuteGpu(
    cudaStream_t stream,
    const int n,
    const void* permuteOrder,
    const void* oldSteps,
    const void* newSteps,
    const void* const inputData,
    void* const outputData)
{
    const int BS = 512;
    const int GS = (n + BS - 1) / BS;

    permuteKernel<BS><<<GS, BS, 0, stream>>>(n,
                                             *(const Quadruple*) permuteOrder,
                                             *(const Quadruple*) oldSteps,
                                             *(const Quadruple*) newSteps,
                                             (const float*) inputData, (float*) outputData);

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

ssdStatus_t permuteInference(
    cudaStream_t stream,
    const bool needPermute,
    const int numAxes,
    const int count,
    const void* permuteOrder,
    const void* oldSteps,
    const void* newSteps,
    const void* const inputData,
    void* const outputData)
{
    if (needPermute)
    {
        assert(numAxes == 4 && "Currently only support 4 dimensions.");
        return permuteGpu(stream, count, permuteOrder, oldSteps, newSteps, inputData, outputData);
    }
    else
    {
        CSC(cudaMemcpyAsync(outputData, inputData, sizeof(float) * count, cudaMemcpyDeviceToDevice, stream), STATUS_FAILURE);
        return STATUS_SUCCESS;
    }
}

} // namespace plugin
} // namespace nvinfer1
