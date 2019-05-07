/*
 * This is the API for YOLO plugins, which supports YOLOv1, YOLOv2 and YOLO9000 in NVIDIA tensorRT.
 * Details of YOLO algorithm can be found here: https://arxiv.org/abs/1612.08242
 * Author has provided the source code of YOLO in darknet: https://pjreddie.com/darknet/yolo/
*/

#ifndef TRT_YOLO_H
#define TRT_YOLO_H
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "plugin.h"
#include <cassert>
#include <iostream>

typedef pluginStatus_t yoloStatus_t;

yoloStatus_t lReLUInference(
    cudaStream_t stream,
    int n,
    float negativeSlope,
    const void* input,
    void* output);

yoloStatus_t reorgInference(
    cudaStream_t stream,
    int batch,
    int C,
    int H,
    int W,
    int stride,
    const void* input,
    void* output);

yoloStatus_t regionInference(
    cudaStream_t stream,
    int batch,
    int C,
    int H,
    int W,
    int num,
    int coords,
    int classes,
    bool hasSoftmaxTree,
    const nvinfer1::plugin::softmaxTree* smTree,
    const void* input,
    void* output);

#endif // TRT_YOLO_H