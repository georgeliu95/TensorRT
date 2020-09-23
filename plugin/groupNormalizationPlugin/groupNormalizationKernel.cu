#include "groupNormalizationPlugin.h"

namespace nvinfer1
{
namespace plugin
{

template <typename T, unsigned TPB>
__global__ void scaleShiftChannelsInplaceKernel(T* inOut, const int ld, const float* beta, const float* gamma)
{
    // grid is blocks x C x B
    // ld should be H*W
    // blockIdx.z = batch
    // blockIdx.y = channel
    // blockIdx.x = block per col
    const T b = beta[blockIdx.y];
    const T g = gamma[blockIdx.y];

    const int offset = (blockIdx.z * gridDim.y + blockIdx.y) * ld;

    const int tx = blockIdx.x * TPB + threadIdx.x;

    if (tx < ld)
    {
        inOut[offset + tx] = g * inOut[offset + tx] + b;
    }
}

template <typename T>
void scaleShiftChannelsInplace(T* inOut, const int B, const int C, const int channelVolume, const float* beta,
    const float* gamma, cudaStream_t stream)
{

    constexpr int TPB = 256;
    const int colBlocks = (channelVolume + TPB - 1) / TPB;
    const dim3 grid(colBlocks, C, B);

    scaleShiftChannelsInplaceKernel<T, TPB><<<grid, TPB, 0, stream>>>(inOut, channelVolume, beta, gamma);

    CUASSERT(cudaPeekAtLastError());
}

template void scaleShiftChannelsInplace<float>(float* inOut, const int B, const int C, const int channelVolume, const float* beta,
    const float* gamma, cudaStream_t stream);
} /* plugin */
} /* nvinfer1 */
