#include "rpnMacros.h"
#include "yolo.h"

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void softmaxKernel(const float* input,
                                  const int n,
                                  const int batch,
                                  const int batchOffset,
                                  const int groups,
                                  const int groupOffset,
                                  const int stride,
                                  const float temp,
                                  float* output)
{
    int id = blockIdx.x * nthdsPerCTA + threadIdx.x;
    if (id < batch * groups)
    {
        int b = id / groups;
        int g = id % groups;
        float sum = 0.;
        float largest = -3.402823466e+38;
        int offset = b * batchOffset + g * groupOffset;
        for (int i = 0; i < n; ++i)
        {
            float val = input[i * stride + offset];
            largest = (val > largest) ? val : largest;
        }
        for (int i = 0; i < n; ++i)
        {
            float e = exp(input[i * stride + offset] / temp - largest / temp);
            sum += e;
            output[i * stride + offset] = e;
        }
        for (int i = 0; i < n; ++i)
            output[i * stride + offset] /= sum;
    }
}

template <unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void activateKernel(float* data,
                                   const int range)
{
    int i = blockIdx.x * nthdsPerCTA + threadIdx.x;
    if (i < range)
        data[i] = 1. / (1. + exp(-data[i]));
}

yoloStatus_t regionGPU(
    cudaStream_t stream,
    const int batch,
    const int C,
    const int H,
    const int W,
    const int num,
    const int coords,
    const int classes,
    const bool hasSoftmaxTree,
    const nvinfer1::plugin::softmaxTree* smTree,
    const float* input,
    float* output)
{
    const int BS = 512;
    const int GS1 = (2 * H * W + BS - 1) / BS;
    const int GS2 = (H * W + BS - 1) / BS;
    for (int b = 0; b < batch; ++b)
    {
        for (int n = 0; n < num; ++n)
        {
            int index = b * C * H * W + n * H * W * (coords + classes + 1);
            activateKernel<BS><<<GS1, BS, 0, stream>>>(output + index, 2 * H * W);
            index = b * C * H * W + n * H * W * (coords + classes + 1) + 4 * H * W;
            activateKernel<BS><<<GS2, BS, 0, stream>>>(output + index, H * W);
        }
    }
    const int GS3 = (batch * num * H * W + BS - 1) / BS;
    if (hasSoftmaxTree)
    {
        int count = 5;
        for (int i = 0; i < smTree->groups; ++i)
        {
            int groupSize = smTree->groupSize[i];
            softmaxKernel<BS><<<GS3, BS, 0, stream>>>(input + count * H * W, groupSize, batch * num, (C * H * W / num), H * W, 1, H * W, 1., output + count * H * W);
            count += groupSize;
        }
    }
    else
    {
        softmaxKernel<BS><<<GS3, BS, 0, stream>>>(input + 5 * H * W, classes, batch * num, (C * H * W / num), H * W, 1, H * W, 1., output + 5 * H * W);
    }

    return STATUS_SUCCESS;
}

yoloStatus_t regionInference(
    cudaStream_t stream,
    const int batch,
    const int C,
    const int H,
    const int W,
    const int num,
    const int coords,
    const int classes,
    const bool hasSoftmaxTree,
    const nvinfer1::plugin::softmaxTree* smTree,
    const void* input,
    void* output)
{
    CHECK(cudaMemcpyAsync(output, input, batch * C * H * W * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    return regionGPU(stream, batch, C, H, W, num, coords, classes, hasSoftmaxTree, smTree, (const float*) input, (float*) output);
}
