#ifndef TRT_CLIP_H
#define TRT_CLIP_H
#include "NvInfer.h"

int clipInference(
    cudaStream_t stream,
    int n,
    float clipMin,
    float clipMax,
    const void* input,
    void* output,
    nvinfer1::DataType type);

#endif // TRT_CLIP_H