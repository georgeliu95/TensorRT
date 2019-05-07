#include "reducedMathPlugin.h"
#include "ssd.h"
#include "ssdMacros.h"
#include <iostream>

namespace nvinfer1
{
namespace plugin
{

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void priorBoxKernel(
        PriorBoxParameters param,
        const int H,
        const int W,
        const int numPriors,
        const int numAspectRatios,
        const float* minSize,
        const float* maxSize,
        const float* aspectRatios,
        float* outputData)
{
    // output dims: (H, W, param.numMinSize, (1+haveMaxSize+numAR-1), 4)
    const int dim = H * W * numPriors;
    const bool haveMaxSize = param.numMaxSize > 0;
    const int dimAR = (haveMaxSize ? 1 : 0) + numAspectRatios;
    //printf(" PriorBox dimAR %d ", dimAR);
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
         i < dim; i += gridDim.x * nthdsPerCTA)
    {
        const int w = (i / numPriors) % W;
        const int h = (i / numPriors) / W;
        const float centerX = (w + param.offset) * param.stepW;
        const float centerY = (h + param.offset) * param.stepH;
        const int minSizeId = (i / dimAR) % param.numMinSize;
        const int arId = i % dimAR;
        if (arId == 0)
        {
            const float boxW = minSize[minSizeId];
            const float boxH = boxW;
            float x, y, z, w;
            x = (centerX - boxW / 2.0f) / param.imgW;
            y = (centerY - boxH / 2.0f) / param.imgH;
            z = (centerX + boxW / 2.0f) / param.imgW;
            w = (centerY + boxH / 2.0f) / param.imgH;
            if (param.clip)
            {
                x = min(max(x, 0.0f), 1.0f);
                y = min(max(y, 0.0f), 1.0f);
                z = min(max(z, 0.0f), 1.0f);
                w = min(max(w, 0.0f), 1.0f);
            }
            outputData[i * 4] = x;
            outputData[i * 4 + 1] = y;
            outputData[i * 4 + 2] = z;
            outputData[i * 4 + 3] = w;
        }
        else if (haveMaxSize && arId == 1)
        {
            const float boxW = sqrt(minSize[minSizeId] * maxSize[minSizeId]);
            const float boxH = boxW;
            float x, y, z, w;
            x = (centerX - boxW / 2.0f) / param.imgW;
            y = (centerY - boxH / 2.0f) / param.imgH;
            z = (centerX + boxW / 2.0f) / param.imgW;
            w = (centerY + boxH / 2.0f) / param.imgH;
            if (param.clip)
            {
                x = min(max(x, 0.0f), 1.0f);
                y = min(max(y, 0.0f), 1.0f);
                z = min(max(z, 0.0f), 1.0f);
                w = min(max(w, 0.0f), 1.0f);
            }
            outputData[i * 4] = x;
            outputData[i * 4 + 1] = y;
            outputData[i * 4 + 2] = z;
            outputData[i * 4 + 3] = w;
        }
        else
        {
            const int arOffset = haveMaxSize ? arId - 1 : arId; // skip aspectRatios[0] which is 1
            const float boxW = minSize[minSizeId] * sqrt(aspectRatios[arOffset]);
            const float boxH = minSize[minSizeId] / sqrt(aspectRatios[arOffset]);
            float x, y, z, w;
            x = (centerX - boxW / 2.0f) / param.imgW;
            y = (centerY - boxH / 2.0f) / param.imgH;
            z = (centerX + boxW / 2.0f) / param.imgW;
            w = (centerY + boxH / 2.0f) / param.imgH;
            if (param.clip)
            {
                x = min(max(x, 0.0f), 1.0f);
                y = min(max(y, 0.0f), 1.0f);
                z = min(max(z, 0.0f), 1.0f);
                w = min(max(w, 0.0f), 1.0f);
            }
            outputData[i * 4] = x;
            outputData[i * 4 + 1] = y;
            outputData[i * 4 + 2] = z;
            outputData[i * 4 + 3] = w;
        }
    }
    float* output = outputData + dim * 4;
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
         i < dim; i += gridDim.x * nthdsPerCTA)
    {
        float x, y, z, w;
        x = param.variance[0];
        y = param.variance[1];
        z = param.variance[2];
        w = param.variance[3];
        output[i * 4] = x;
        output[i * 4 + 1] = y;
        output[i * 4 + 2] = z;
        output[i * 4 + 3] = w;
    }
}

ssdStatus_t priorBoxGpu(
    cudaStream_t stream,
    const PriorBoxParameters param,
    const int H,
    const int W,
    const int numPriors,
    const int numAspectRatios,
    const void* minSize,
    const void* maxSize,
    const void* aspectRatios,
    void* outputData)
{
    //assert((uintptr_t)outputData & 127 == 0);
    const int dim = H * W * numPriors;
    if (dim > 5120)
    {
        const int BS = 128;
        const int GS = (dim + BS - 1) / BS;
        priorBoxKernel<BS><<<GS, BS, 0, stream>>>(param, H, W, numPriors, numAspectRatios,
                                                  (const float*) minSize, (const float*) maxSize,
                                                  (const float*) aspectRatios, (float*) outputData);
        CSC(cudaGetLastError(), STATUS_FAILURE);
        return STATUS_SUCCESS;
    }
    else
    {
        const int BS = 32;
        const int GS = (dim + BS - 1) / BS;
        priorBoxKernel<BS><<<GS, BS, 0, stream>>>(param, H, W, numPriors, numAspectRatios,
                                                  (const float*) minSize, (const float*) maxSize,
                                                  (const float*) aspectRatios, (float*) outputData);
        CSC(cudaGetLastError(), STATUS_FAILURE);
        return STATUS_SUCCESS;
    }
}

ssdStatus_t priorBoxInference(
    cudaStream_t stream,
    const PriorBoxParameters param,
    const int H,
    const int W,
    const int numPriors,
    const int numAspectRatios,
    const void* minSize,
    const void* maxSize,
    const void* aspectRatios,
    void* outputData)
{
    assert(param.numMaxSize >= 0);
    if (param.numMaxSize)
        return priorBoxGpu(stream, param, H, W, numPriors, numAspectRatios,
                           minSize, maxSize, aspectRatios, outputData);
    else
        return priorBoxGpu(stream, param, H, W, numPriors, numAspectRatios,
                           minSize, nullptr, aspectRatios, outputData);
}

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void gridAnchorKernel(
        const GridAnchorParameters param,
        const int numAspectRatios,
        reduced_divisor divObj,
        const float* widths,
        const float* heights,
        float* outputData)
{
    // output dims: (H, W, param.numMinSize, (1+haveMaxSize+numAR-1), 4)
    const int dim = param.H * param.W * numAspectRatios;
    float anchorStride = (1.0 / param.H);
    float anchorOffset = 0.5 * anchorStride;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dim)
        return;
    int arId, currIndex;
    divObj.divmod(tid, currIndex, arId);

    const int w = currIndex % param.W;
    const int h = currIndex / param.W;

    float yC = h * anchorStride + anchorOffset;
    float xC = w * anchorStride + anchorOffset;

    float xMin = xC - 0.5 * widths[arId];
    float yMin = yC - 0.5 * heights[arId];

    float xMax = xC + 0.5 * widths[arId];
    float yMax = yC + 0.5 * heights[arId];

    outputData[tid * 4] = xMin;
    outputData[tid * 4 + 1] = yMin;
    outputData[tid * 4 + 2] = xMax;
    outputData[tid * 4 + 3] = yMax;

    float* output = outputData + dim * 4;

    output[tid * 4] = param.variance[0];
    output[tid * 4 + 1] = param.variance[1];
    output[tid * 4 + 2] = param.variance[2];
    output[tid * 4 + 3] = param.variance[3];
}

ssdStatus_t anchorGridInference(
    cudaStream_t stream,
    const GridAnchorParameters param,
    const int numAspectRatios,
    const void* widths,
    const void* heights,
    void* outputData)
{
    const int dim = param.H * param.W * numAspectRatios;
    reduced_divisor divObj(numAspectRatios);
    if (dim > 5120)
    {
        const int BS = 128;
        const int GS = (dim + BS - 1) / BS;
        gridAnchorKernel<BS><<<GS, BS, 0, stream>>>(param, numAspectRatios, divObj,
                                                    (const float*) widths, (const float*) heights,
                                                    (float*) outputData);
    }
    else
    {
        const int BS = 32;
        const int GS = (dim + BS - 1) / BS;
        gridAnchorKernel<BS><<<GS, BS, 0, stream>>>(param, numAspectRatios, divObj,
                                                    (const float*) widths, (const float*) heights,
                                                    (float*) outputData);
    }
    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}

} // namespace plugin
} // namespace nvinfer1
