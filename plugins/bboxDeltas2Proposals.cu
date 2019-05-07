#include "rpnMacros.h"
#include "rpnlayer.h"
#include "rpnlayer_internal.h"
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

using std::max;
using std::min;

// BBD2P KERNEL {{{
template <typename T_DELTAS,
          DLayout_t L_DELTAS,
          typename TV_PROPOSALS,
          DLayout_t L_PROPOSALS,
          typename T_FGSCORES,
          DLayout_t L_FGSCORES>
__global__ void bboxDeltas2Proposals_kernel(
    int N,
    int A,
    int H,
    int W,
    const float* __restrict__ anchors,
    const float* __restrict__ imInfo,
    int featureStride,
    float minSize,
    const T_DELTAS* __restrict__ deltas,
    TV_PROPOSALS* __restrict__ proposals,
    T_FGSCORES* __restrict__ scores)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N * A * H * W)
    { // TODO this can be a loop.
        int cnt = tid;
        int w = cnt % W;
        cnt = cnt / W;
        int h = cnt % H;
        cnt = cnt / H;
        int a = cnt % A;
        cnt = cnt / A;
        int n = cnt;
        int hw = h * W + w;

        float imHeight = imInfo[3 * n];
        float imWidth = imInfo[3 * n + 1];

        float4 anchor = ((float4*) anchors)[a];
        float a_ctr_x = anchor.x;
        float a_ctr_y = anchor.y;
        float a_w = anchor.z;
        float a_h = anchor.w;

        int id = ((tid - hw) * 4) + hw;
        T_DELTAS dx;
        T_DELTAS dy;
        T_DELTAS dw;
        T_DELTAS dh;
        if (L_DELTAS == NCHW)
        {
            dx = deltas[id];
            dy = deltas[id + 1 * H * W];
            dw = deltas[id + 2 * H * W];
            dh = deltas[id + 3 * H * W];
        }
        else if (L_DELTAS == NC4HW)
        {
            dx = deltas[tid * 4 + 0];
            dy = deltas[tid * 4 + 1];
            dw = deltas[tid * 4 + 2];
            dh = deltas[tid * 4 + 3];
        }
        float ctr_x = a_ctr_x + w * featureStride;
        float ctr_y = a_ctr_y + h * featureStride;
        ctr_x = ctr_x + dx * a_w;
        ctr_y = ctr_y + dy * a_h;
        float b_w = __expf(dw) * a_w;
        float b_h = __expf(dh) * a_h;
        float bx = ctr_x - (b_w / 2);
        float by = ctr_y - (b_h / 2);
        float bz = ctr_x + (b_w / 2);
        float bw = ctr_y + (b_h / 2);

        TV_PROPOSALS bbox;
        bbox.x = fminf(fmaxf(bx, 0.0f), imWidth - 1.0f);
        bbox.y = fminf(fmaxf(by, 0.0f), imHeight - 1.0f);
        bbox.z = fminf(fmaxf(bz, 0.0f), imWidth - 1.0f);
        bbox.w = fminf(fmaxf(bw, 0.0f), imHeight - 1.0f);

        if (L_PROPOSALS == NC4HW)
        {
            proposals[tid] = bbox;
        }

        int ininf = 0xff800000;
        float ninf = *(float*) &ininf;
        float scaledMinSize = minSize * imInfo[3 * n + 2];
        if (bbox.z - bbox.x + 1 < scaledMinSize || bbox.w - bbox.y + 1 < scaledMinSize)
        {
            if (L_FGSCORES == NCHW)
                scores[tid] = ninf;
        }
    }
}
// }}}

// BBD2P KERNEL LAUNCHER {{{
template <typename T_DELTAS,
          DLayout_t L_DELTAS,
          typename TV_PROPOSALS,
          DLayout_t L_PROPOSALS,
          typename T_FGSCORES,
          DLayout_t L_FGSCORES>
frcnnStatus_t bboxDeltas2Proposals_gpu(cudaStream_t stream,
                                       int N,
                                       int A,
                                       int H,
                                       int W,
                                       const float* imInfo,
                                       int featureStride,
                                       float minBoxSize,
                                       const float* anchors,
                                       const void* deltas,
                                       void* propos,
                                       void* scores)
{
    const int BS = 32;
    const int GS = ((N * A * H * W) + BS - 1) / BS;

    bboxDeltas2Proposals_kernel<T_DELTAS, L_DELTAS, TV_PROPOSALS, L_PROPOSALS, T_FGSCORES, L_FGSCORES><<<GS, BS, 0, stream>>>(N, A, H, W,
                                                                                                                              anchors,
                                                                                                                              imInfo,
                                                                                                                              featureStride,
                                                                                                                              minBoxSize,
                                                                                                                              (T_DELTAS*) deltas,
                                                                                                                              (TV_PROPOSALS*) propos,
                                                                                                                              (T_FGSCORES*) scores);

    DEBUG_PRINTF("&&&& [bboxD2P] POST LAUNCH\n");
    DEBUG_PRINTF("&&&& [bboxD2P] PROPOS %u\n", hash(propos, N * A * H * W * 4 * sizeof(float)));

    CSC(cudaGetLastError(), STATUS_FAILURE);
    return STATUS_SUCCESS;
}
// }}}

// BBD2P LAUNCH CONFIG {{{
typedef frcnnStatus_t (*bd2pFun)(cudaStream_t,
                                 int,
                                 int,
                                 int,
                                 int,
                                 const float*,
                                 int,
                                 float,
                                 const float*,
                                 const void*,
                                 void*,
                                 void*);

struct bd2pLaunchConfig
{
    DType_t t_deltas;
    DLayout_t l_deltas;
    DType_t t_proposals;
    DLayout_t l_proposals;
    DType_t t_scores;
    DLayout_t l_scores;
    bd2pFun function;

    bd2pLaunchConfig(DType_t t_deltas, DLayout_t l_deltas, DType_t t_proposals, DLayout_t l_proposals, DType_t t_scores, DLayout_t l_scores)
        : t_deltas(t_deltas)
        , l_deltas(l_deltas)
        , t_proposals(t_proposals)
        , l_proposals(l_proposals)
        , t_scores(t_scores)
        , l_scores(l_scores)
    {
    }

    bd2pLaunchConfig(DType_t t_deltas, DLayout_t l_deltas, DType_t t_proposals, DLayout_t l_proposals, DType_t t_scores, DLayout_t l_scores, bd2pFun function)
        : t_deltas(t_deltas)
        , l_deltas(l_deltas)
        , t_proposals(t_proposals)
        , l_proposals(l_proposals)
        , t_scores(t_scores)
        , l_scores(l_scores)
        , function(function)
    {
    }

    bool operator==(const bd2pLaunchConfig& other)
    {
        return t_deltas == other.t_deltas && l_deltas == other.l_deltas && t_proposals == other.t_proposals && l_proposals == other.l_proposals && t_scores == other.t_scores && l_scores == other.l_scores;
    }
};

static std::vector<bd2pLaunchConfig> bd2pFunVec;
#define FLOAT32 nvinfer1::DataType::kFLOAT
bool init()
{
    bd2pFunVec.push_back(bd2pLaunchConfig(FLOAT32, NC4HW,
                                          FLOAT32, NC4HW,
                                          FLOAT32, NCHW,
                                          bboxDeltas2Proposals_gpu<float, NC4HW, float4, NC4HW, float, NCHW>));
    bd2pFunVec.push_back(bd2pLaunchConfig(FLOAT32, NCHW,
                                          FLOAT32, NC4HW,
                                          FLOAT32, NCHW,
                                          bboxDeltas2Proposals_gpu<float, NCHW, float4, NC4HW, float, NCHW>));
    return true;
}
static bool initialized = init();
// }}}

// BBD2P {{{
frcnnStatus_t bboxDeltas2Proposals(cudaStream_t stream,
                                   const int N,
                                   const int A,
                                   const int H,
                                   const int W,
                                   const int featureStride,
                                   const float minBoxSize,
                                   const float* imInfo,
                                   const float* anchors,
                                   const DType_t t_deltas,
                                   const DLayout_t l_deltas,
                                   const void* deltas,
                                   const DType_t t_proposals,
                                   const DLayout_t l_proposals,
                                   void* proposals,
                                   const DType_t t_scores,
                                   const DLayout_t l_scores,
                                   void* scores)
{
    bd2pLaunchConfig lc = bd2pLaunchConfig(t_deltas, l_deltas, t_proposals, l_proposals, t_scores, l_scores);
    for (unsigned i = 0; i < bd2pFunVec.size(); i++)
    {
        if (lc == bd2pFunVec[i])
        {
            DEBUG_PRINTF("BBD2P kernel %d\n", i);
            return bd2pFunVec[i].function(stream,
                                          N, A, H, W,
                                          imInfo,
                                          featureStride,
                                          minBoxSize,
                                          anchors,
                                          deltas,
                                          proposals,
                                          scores);
        }
    }
    return STATUS_BAD_PARAM;
}
// }}}
