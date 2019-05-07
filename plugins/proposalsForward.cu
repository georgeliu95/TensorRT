#include "rpnMacros.h"
#include "rpnlayer.h"
#include "rpnlayer_internal.h"

// PROPOSALS INFERENCE {{{
frcnnStatus_t proposalsInference(cudaStream_t stream,
                                 const int N,
                                 const int A,
                                 const int H,
                                 const int W,
                                 const int featureStride,
                                 const int preNmsTop,
                                 const int nmsMaxOut,
                                 const float iouThreshold,
                                 const float minBoxSize,
                                 const float* imInfo,
                                 const float* anchors,
                                 const DType_t t_scores,
                                 const DLayout_t l_scores,
                                 const void* scores,
                                 const DType_t t_deltas,
                                 const DLayout_t l_deltas,
                                 const void* deltas,
                                 void* workspace,
                                 const DType_t t_rois,
                                 void* rois)
{
    if (imInfo == NULL || anchors == NULL || scores == NULL || deltas == NULL || workspace == NULL || rois == NULL)
    {
        return STATUS_BAD_PARAM;
    }

    DEBUG_PRINTF("&&&& IM INFO %u\n", hash(imInfo, N * 3 * sizeof(float)));
    DEBUG_PRINTF("&&&& ANCHORS %u\n", hash(anchors, 9 * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& SCORES  %u\n", hash(scores, N * A * 2 * H * W * sizeof(float)));
    DEBUG_PRINTF("&&&& DELTAS  %u\n", hash(deltas, N * A * 4 * H * W * sizeof(float)));

    size_t nmsWorkspaceSize = proposalsForwardNMSWorkspaceSize(N, A, H, W, nmsMaxOut);
    void* nmsWorkspace = workspace;

    size_t proposalsSize = proposalsForwardBboxWorkspaceSize(N, A, H, W);
    const DType_t t_proposals = nvinfer1::DataType::kFLOAT;
    const DLayout_t l_proposals = NC4HW;
    void* proposals = nextWorkspacePtr((int8_t*) nmsWorkspace, nmsWorkspaceSize);

    const DType_t t_fgScores = t_scores;
    const DLayout_t l_fgScores = NCHW;
    void* fgScores = nextWorkspacePtr((int8_t*) proposals, proposalsSize);

    frcnnStatus_t status;

    status = extractFgScores(stream,
                             N, A, H, W,
                             t_scores, l_scores, scores,
                             t_fgScores, l_fgScores, fgScores);
    FRCNN_ASSERT_FAILURE(status == STATUS_SUCCESS);

    DEBUG_PRINTF("&&&& FG SCORES %u\n", hash((void*) fgScores, N * A * H * W * sizeof(float)));
    DEBUG_PRINTF("&&&& DELTAS %u\n", hash((void*) proposals, N * A * H * W * 4 * sizeof(float)));

    status = bboxDeltas2Proposals(stream,
                                  N, A, H, W,
                                  featureStride,
                                  minBoxSize,
                                  imInfo,
                                  anchors,
                                  t_deltas, l_deltas, deltas,
                                  t_proposals, l_proposals, proposals,
                                  t_fgScores, l_fgScores, fgScores);
    FRCNN_ASSERT_FAILURE(status == STATUS_SUCCESS);

    DEBUG_PRINTF("&&&& PROPOSALS %u\n", hash((void*) proposals, N * A * H * W * 4 * sizeof(float)));
    DEBUG_PRINTF("&&&& FG SCORES %u\n", hash((void*) fgScores, N * A * H * W * sizeof(float)));

    status = nms(stream,
                 N,
                 A * H * W,
                 preNmsTop,
                 nmsMaxOut,
                 iouThreshold,
                 t_fgScores, l_fgScores, fgScores,
                 t_proposals, l_proposals, proposals,
                 nmsWorkspace,
                 t_rois, rois);
    FRCNN_ASSERT_FAILURE(status == STATUS_SUCCESS);

    DEBUG_PRINTF("&&&& ROIS %u\n", hash((void*) rois, N * nmsMaxOut * 4 * sizeof(float)));

    return STATUS_SUCCESS;
}
// }}}

// WORKSPACE SIZES {{{
size_t proposalsForwardNMSWorkspaceSize(int N,
                                        int A,
                                        int H,
                                        int W,
                                        int nmsMaxOut)
{
    return N * A * H * W * 5 * 5 * sizeof(float) + (1 << 22);
}

size_t proposalsForwardBboxWorkspaceSize(int N,
                                         int A,
                                         int H,
                                         int W)
{
    return N * A * H * W * 4 * sizeof(float);
}
size_t proposalForwardFgScoresWorkspaceSize(int N,
                                            int A,
                                            int H,
                                            int W)
{
    return N * A * H * W * sizeof(float);
}

size_t proposalsInferenceWorkspaceSize(int N,
                                       int A,
                                       int H,
                                       int W,
                                       int nmsMaxOut)
{
    size_t wss[3];
    wss[0] = proposalsForwardNMSWorkspaceSize(N, A, H, W, nmsMaxOut);
    wss[1] = proposalsForwardBboxWorkspaceSize(N, A, H, W);
    wss[2] = proposalForwardFgScoresWorkspaceSize(N, A, H, W);
    return calculateTotalWorkspaceSize(wss, 3);
}

// }}}
