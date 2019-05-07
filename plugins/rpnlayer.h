#ifndef TRT_RPNLAYER_H
#define TRT_RPNLAYER_H
#include "NvInfer.h"
#include "plugin.h"
#include <cstddef>

//#pragma message "RPN LAYER H"

typedef nvinfer1::DataType DType_t;

typedef enum {
    NCHW = 0,
    NC4HW = 1
} DLayout_t;

// GENERATE ANCHORS {{{
// For now it takes host pointers - ratios and scales but
// in GPU MODE anchors should be device pointer
frcnnStatus_t generateAnchors(cudaStream_t stream,
                              int numRatios,   // number of ratios
                              float* ratios,   // ratio array
                              int numScales,   // number of scales
                              float* scales,   // scale array
                              int baseSize,    // size of the base anchor (baseSize x baseSize)
                              float* anchors); // output anchors (numRatios x numScales)
//}}}

// BBD2P {{{
frcnnStatus_t bboxDeltas2Proposals(cudaStream_t stream,
                                   int N,                // batch size
                                   int A,                // number of anchors
                                   int H,                // last feature map H
                                   int W,                // last feature map W
                                   int featureStride,    // feature stride
                                   float minBoxSize,     // minimum allowed box size before scaling
                                   const float* imInfo,        // image info (nrows, ncols, image scale)
                                   const float* anchors,       // input anchors
                                   DType_t tDeltas,      // type of input deltas
                                   DLayout_t lDeltas,    // layout of input deltas
                                   const void* deltas,         // input deltas
                                   DType_t tProposals,   // type of output proposals
                                   DLayout_t lProposals, // layout of output proposals
                                   void* proposals,            // output proposals
                                   DType_t tScores,      // type of output scores
                                   DLayout_t lScores,    // layout of output scores
                                   void* scores);              // output scores (the score associated with too small box will be set to -inf)
// }}}

// NMS {{{
frcnnStatus_t nms(cudaStream_t stream,
                  int N,                // batch size
                  int R,                // number of ROIs (region of interest) per image
                  int preNmsTop,        // number of proposals before applying NMS
                  int nmsMaxOut,        // number of remaining proposals after applying NMS
                  float iouThreshold,   // IoU threshold
                  DType_t tFgScores,    // type of foreground scores
                  DLayout_t lFgScores,  // layout of foreground scores
                  void* fgScores,             // foreground scores
                  DType_t tProposals,   // type of proposals
                  DLayout_t lProposals, // layout of proposals
                  const void* proposals,      // proposals
                  void* workspace,            // workspace
                  DType_t tRois,        // type of ROIs
                  void* rois);                // ROIs
// }}}

// WORKSPACE SIZES {{{
size_t proposalsForwardNMSWorkspaceSize(int N,
                                        int A,
                                        int H,
                                        int W,
                                        int nmsMaxOut);

size_t proposalsForwardBboxWorkspaceSize(int N,
                                         int A,
                                         int H,
                                         int W);

size_t proposalForwardFgScoresWorkspaceSize(int N,
                                            int A,
                                            int H,
                                            int W);

size_t proposalsInferenceWorkspaceSize(int N,
                                       int A,
                                       int H,
                                       int W,
                                       int nmsMaxOut);

size_t RPROIInferenceFusedWorkspaceSize(int N,
                                        int A,
                                        int H,
                                        int W,
                                        int nmsMaxOut);
// }}}

// PROPOSALS INFERENCE {{{
frcnnStatus_t proposalsInference(cudaStream_t stream,
                                 int N,
                                 int A,
                                 int H,
                                 int W,
                                 int featureStride,
                                 int preNmsTop,
                                 int nmsMaxOut,
                                 float iouThreshold,
                                 float minBoxSize,
                                 const float* imInfo,
                                 const float* anchors,
                                 DType_t tScores,
                                 DLayout_t lScores,
                                 const void* scores,
                                 DType_t tDeltas,
                                 DLayout_t lDeltas,
                                 const void* deltas,
                                 void* workspace,
                                 DType_t tRois,
                                 void* rois);
// }}}

// EXTRACT FG SCORES {{{
frcnnStatus_t extractFgScores(cudaStream_t stream,
                              int N,
                              int A,
                              int H,
                              int W,
                              DType_t tScores,
                              DLayout_t lScores,
                              const void* scores,
                              DType_t tFgScores,
                              DLayout_t lFgScores,
                              void* fgScores);
// }}}

// ROI INFERENCE {{{
frcnnStatus_t roiInference(cudaStream_t stream,
                           int R,        // TOTAL number of rois -> ~nmsMaxOut * N
                           int N,        // Batch size
                           int C,        // Channels
                           int H,        // Input feature map H
                           int W,        // Input feature map W
                           int poolingH, // Output feature map H
                           int poolingW, // Output feature map W
                           float spatialScale,
                           DType_t tRois,
                           const void* rois,
                           DType_t tFeatureMap,
                           DLayout_t lFeatureMap,
                           const void* featureMap,
                           DType_t tTop,
                           DLayout_t lTop,
                           void* top);
// }}}

// ROI FORWARD {{{
frcnnStatus_t roiForward(cudaStream_t stream,
                         int R,        // TOTAL number of rois -> ~nmsMaxOut * N
                         int N,        // Batch size
                         int C,        // Channels
                         int H,        // Input feature map H
                         int W,        // Input feature map W
                         int poolingH, // Output feature map H
                         int poolingW, // Output feature map W
                         float spatialScale,
                         DType_t tRois,
                         const void* rois,
                         DType_t tFeatureMap,
                         DLayout_t lFeatureMap,
                         const void* featureMap,
                         DType_t tTop,
                         DLayout_t lTop,
                         void* top,
                         int* maxIds);
// }}}

// RP ROI Fused INFERENCE {{{
frcnnStatus_t RPROIInferenceFused(cudaStream_t stream,
                                  int N,
                                  int A,
                                  int C,
                                  int H,
                                  int W,
                                  int poolingH,
                                  int poolingW,
                                  int featureStride,
                                  int preNmsTop,
                                  int nmsMaxOut,
                                  float iouThreshold,
                                  float minBoxSize,
                                  float spatialScale,
                                  const float* imInfo,
                                  const float* anchors,
                                  DType_t tScores,
                                  DLayout_t lScores,
                                  const void* scores,
                                  DType_t tDeltas,
                                  DLayout_t lDeltas,
                                  const void* deltas,
                                  DType_t tFeatureMap,
                                  DLayout_t lFeatureMap,
                                  const void* featureMap,
                                  void* workspace,
                                  DType_t tRois,
                                  void* rois,
                                  DType_t tTop,
                                  DLayout_t lTop,
                                  void* top);
// }}}

// GENERATE ANCHORS CPU {{{
frcnnStatus_t generateAnchors_cpu(int numRatios,
                                  float* ratios,
                                  int numScales,
                                  float* scales,
                                  int baseSize,
                                  float* anchors);
// }}}

#endif // TRT_RPNLAYER_H