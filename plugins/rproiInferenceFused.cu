#include "rpnMacros.h"
#include "rpnlayer.h"
#include "rpnlayer_internal.h"

frcnnStatus_t RPROIInferenceFused(cudaStream_t stream,
                                  const int N,
                                  const int A,
                                  const int C,
                                  const int H,
                                  const int W,
                                  const int poolingH,
                                  const int poolingW,
                                  const int featureStride,
                                  const int preNmsTop,
                                  const int nmsMaxOut,
                                  const float iouThreshold,
                                  const float minBoxSize,
                                  const float spatialScale,
                                  const float* imInfo,
                                  const float* anchors,
                                  const DType_t t_scores,
                                  const DLayout_t l_scores,
                                  const void* scores,
                                  const DType_t t_deltas,
                                  const DLayout_t l_deltas,
                                  const void* deltas,
                                  const DType_t t_featureMap,
                                  const DLayout_t l_featureMap,
                                  const void* featureMap,
                                  void* workspaces,
                                  const DType_t t_rois,
                                  void* rois,
                                  const DType_t t_top,
                                  const DLayout_t l_top,
                                  void* top)
{
    if (imInfo == NULL || anchors == NULL || scores == NULL || deltas == NULL || featureMap == NULL || workspaces == NULL || rois == NULL || top == NULL)
    {
        return STATUS_BAD_PARAM;
    }

    frcnnStatus_t status;
    status = proposalsInference(stream,
                                N,
                                A,
                                H,
                                W,
                                featureStride,
                                preNmsTop,
                                nmsMaxOut,
                                iouThreshold,
                                minBoxSize,
                                imInfo,
                                anchors,
                                t_scores,
                                l_scores,
                                scores,
                                t_deltas,
                                l_deltas,
                                deltas,
                                workspaces,
                                t_rois,
                                rois);

    FRCNN_ASSERT_FAILURE(status == STATUS_SUCCESS);

    status = roiInference(stream,
                          N * nmsMaxOut, // TOTAL number of rois -> ~nmsMaxOut * N
                          N,             // Batch size
                          C,             // Channels
                          H,             // Input feature map H
                          W,             // Input feature map W
                          poolingH,      // Output feature map H
                          poolingW,      // Output feature map W
                          spatialScale,
                          t_rois,
                          rois,
                          t_featureMap,
                          l_featureMap,
                          featureMap,
                          t_top,
                          l_top,
                          top);

    FRCNN_ASSERT_FAILURE(status == STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

size_t RPROIInferenceFusedWorkspaceSize(int N,
                                        int A,
                                        int H,
                                        int W,
                                        int nmsMaxOut)
{
    return proposalsInferenceWorkspaceSize(N, A, H, W, nmsMaxOut);
}
