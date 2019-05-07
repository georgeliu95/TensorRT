#include "rpnMacros.h"
#include "rpnlayer.h"
#include "rpnlayer_internal.h"

#define FLOAT32 nvinfer1::DataType::kFLOAT

template <typename T>
frcnnStatus_t extractFgScores_gpu(cudaStream_t stream,
                                  int N,
                                  int A,
                                  int H,
                                  int W,
                                  const void* scores,
                                  void* fgScores)
{
    //TODO custom kernel for this
    size_t size = A * H * W * sizeof(T);
    for (int n = 0; n < N; n++)
    {
        size_t offset_ld = (n * 2 + 1) * A * H * W;
        size_t offset_st = n * A * H * W;
        CSC(cudaMemcpyAsync(((T*) fgScores) + offset_st, ((T*) scores) + offset_ld, size, cudaMemcpyDeviceToDevice, stream), STATUS_FAILURE);
    }

    return STATUS_SUCCESS;
}

template <typename T>
frcnnStatus_t extractFgScores_cpu(int N,
                                  int A,
                                  int H,
                                  int W,
                                  const void* scores,
                                  void* fgScores)
{
    size_t size = A * H * W * sizeof(T);
    for (int n = 0; n < N; n++)
    {
        size_t offset_ld = (n * 2 + 1) * A * H * W;
        size_t offset_st = n * A * H * W;
        memcpy(((T*) fgScores) + offset_st, ((T*) scores) + offset_ld, size);
    }
    return STATUS_SUCCESS;
}

frcnnStatus_t extractFgScores(cudaStream_t stream,
                              const int N,
                              const int A,
                              const int H,
                              const int W,
                              const DType_t t_scores,
                              const DLayout_t l_scores,
                              const void* scores,
                              const DType_t t_fgScores,
                              const DLayout_t l_fgScores,
                              void* fgScores)
{
    if (l_fgScores != NCHW || l_scores != NCHW)
        return STATUS_BAD_PARAM;

    if (t_fgScores != FLOAT32)
        return STATUS_BAD_PARAM;

    if (t_scores != FLOAT32)
        return STATUS_BAD_PARAM;

    return extractFgScores_gpu<float>(stream, N, A, H, W, scores, fgScores);
}
