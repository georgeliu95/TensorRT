#include "ssdMacros.h"
#include "ssd_internal.h"

#define CUBLAS_CHECK(condition)                                                                 \
    do                                                                                          \
    {                                                                                           \
        cublasStatus_t status = condition;                                                      \
        if (status != CUBLAS_STATUS_SUCCESS)                                                    \
        {                                                                                       \
            printf("%s %d CUBLAS FAIL %s\n", __FILE__, __LINE__, cublasGetErrorString(status)); \
        }                                                                                       \
    } while (0)

namespace nvinfer1
{
namespace plugin
{

size_t normalizePluginWorkspaceSize(bool acrossSpatial, int C, int H, int W)
{
    if (acrossSpatial)
        return sizeof(float) * C * H * W;
    else
        return (size_t) 0;
}

template <unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta)
    __global__ void normalizeNotAcrossSpatialKernel(
        const bool channelShared,
        const int N,
        const int C,
        const int H,
        const int W,
        const float eps,
        const float* scale,
        float* inputData,
        float* outputData)
{
    const int dim = C * H * W;
    const int spatialDim = H * W;
    const int tile = 32;
    const int numTile = (spatialDim + tile - 1) / tile;
    for (int n = blockIdx.x; n < N * numTile; n += gridDim.x)
    {
        float* input = inputData + (n / numTile) * dim;
        float* output = outputData + (n / numTile) * dim;
        __shared__ float sum[tile];
        float localsum = 0.0f;
        for (int i = threadIdx.x; i < tile; i += nthds_per_cta)
        {
            sum[i] = 0.0f;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta)
        {
            int row = i / tile;
            int col = (n % numTile) * tile + i % tile;
            float data = 0.0f;
            if (col < spatialDim)
                data = input[row * spatialDim + col];
            localsum += data * data;
        }
        atomicAdd(&sum[threadIdx.x & 31], localsum);
        __syncthreads();
        for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta)
        {
            int row = i / tile;
            int col = (n % numTile) * tile + i % tile;
            if (col < spatialDim)
            {
                int offset = row * spatialDim + col;
                output[offset] = input[offset] / sqrt(sum[threadIdx.x & 31] + eps);
            }
        }
        if (channelShared)
        {
            for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta)
            {
                int row = i / tile;
                int col = (n % numTile) * tile + i % tile;
                if (col < spatialDim)
                    output[row * spatialDim + col] *= scale[0];
            }
        }
        else
        {
            for (int i = threadIdx.x; i < C * tile; i += nthds_per_cta)
            {
                int row = i / tile;
                int col = (n % numTile) * tile + i % tile;
                if (col < spatialDim)
                    output[row * spatialDim + col] *= scale[row];
            }
        }
    }
}

ssdStatus_t normalizeNotAcrossSpatialGpu(
    cudaStream_t stream,
    const bool channelShared,
    const int N,
    const int C,
    const int H,
    const int W,
    const float eps,
    const void* scale,
    const void* inputData,
    void* outputData)
{
    const int BS = 128;
    const int GS = 256;
    // assumes warp size == 32
    assert(BS % 32 == 0);
    normalizeNotAcrossSpatialKernel<BS><<<GS, BS, 0, stream>>>(channelShared, N, C, H, W, eps,
                                                               (const float*) scale,
                                                               (float*) inputData,
                                                               (float*) outputData);
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

__global__ void squareKernel(
    const int n,
    const float* x,
    float* y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        y[i] = x[i] * x[i];
    }
}

__global__ void scalChannelKernel(
    const int n,
    const int spatialDim,
    const float* inputData,
    const float* scale,
    float* outputData)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += gridDim.x * blockDim.x)
    {
        outputData[i] = inputData[i] / scale[i / spatialDim];
    }
}

ssdStatus_t normalizeInference(
    cudaStream_t stream,
    cublasHandle_t handle,
    const bool acrossSpatial,
    const bool channelShared,
    const int N,
    const int C,
    const int H,
    const int W,
    const float eps,
    const void* scale,
    const void* inputData,
    void* outputData,
    void* workspace)
{
    const int dim = C * H * W;
    if (acrossSpatial)
    {
        float* input = (float*) const_cast<void*>(inputData);
        float* output = (float*) outputData;
        float* buffer = (float*) workspace;
        for (int n = 0; n < N; ++n)
        {
            squareKernel<<<(dim + 511) / 512, 512, 0, stream>>>(dim, input, buffer);
            float normsqr = 0.0f;
            CUBLAS_CHECK(cublasSasum(handle, dim, buffer, 1, &normsqr));
            CUBLAS_CHECK(cublasScopy(handle, dim, input, 1, output, 1));
            normsqr = 1 / sqrt(normsqr + eps);
            CUBLAS_CHECK(cublasSscal(handle, dim, &normsqr, output, 1));
            if (channelShared)
            {
                CUBLAS_CHECK(cublasSscal(handle, dim, (float*) scale, output, 1));
            }
            else
            {
                scalChannelKernel<<<(dim + 511) / 512, 512, 0, stream>>>(dim, H * W, output, (float*) scale, output);
            }
            input += dim;
            output += dim;
        }
        return STATUS_SUCCESS;
    }
    else
    {
        return normalizeNotAcrossSpatialGpu(stream, channelShared, N, C, H, W, eps, scale, inputData, outputData);
    }
}

} // namespace plugin
} // namespace nvinfer1
