#include "yolo.h"

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void pReLUKernel(
        const int n,
        const float negativeSlope,
        const float* input,
        float* output)
{
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = input[i] > 0 ? input[i] : input[i] * negativeSlope;
    }
}

yoloStatus_t lReLUGPU(
    cudaStream_t stream,
    const int n,
    const float negativeSlope,
    const void* input,
    void* output)
{
    const int BS = 512;
    const int GS = (n + BS - 1) / BS;
    pReLUKernel<BS><<<GS, BS, 0, stream>>>(n, negativeSlope,
                                           (const float*) input,
                                           (float*) output);
    return STATUS_SUCCESS;
}

yoloStatus_t lReLUInference(
    cudaStream_t stream,
    const int n,
    const float negativeSlope,
    const void* input,
    void* output)
{
    return lReLUGPU(stream, n, negativeSlope, (const float*) input, (float*) output);
}
