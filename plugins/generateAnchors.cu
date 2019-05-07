#include "rpnMacros.h"
#include "rpnlayer.h"
#include "rpnlayer_internal.h"
#include <cstdio>

frcnnStatus_t generateAnchors_cpu(int numRatios,
                                  float* ratios,
                                  int numScales,
                                  float* scales,
                                  int baseSize,
                                  float* anchors)
{
#ifdef DEBUG
    DEBUG_PRINTF("Generating Anchors with:\n");
    DEBUG_PRINTF("Scales:");
    for (int s = 0; s < numScales; ++s)
    {
        DEBUG_PRINTF("%f\t", scales[s]);
    }
    DEBUG_PRINTF("\n");
    DEBUG_PRINTF("Ratios:");
    for (int r = 0; r < numRatios; ++r)
    {
        DEBUG_PRINTF("%f\t", ratios[r]);
    }
    DEBUG_PRINTF("\n");
#endif

    if ((numScales <= 0) || (numRatios <= 0) || (baseSize <= 0))
    {
        return STATUS_BAD_PARAM;
    }

    for (int r = 0; r < numRatios; ++r)
    {
        for (int s = 0; s < numScales; ++s)
        {
            int id = r * numScales + s;
            float scale = scales[s];
            float ratio = ratios[r];
            float bs = baseSize;
            float ws = round(sqrt((float) (bs * bs) / ratio));
            float hs = round(ws * ratio);
            ws *= scale;
            hs *= scale;

            anchors[id * 4] = (bs - 1) / 2;
            anchors[id * 4 + 1] = (bs - 1) / 2;
            anchors[id * 4 + 2] = ws;
            anchors[id * 4 + 3] = hs;
            /*
            DEBUG_PRINTF("%d %d %d %d\n",
                   (int)((bs / 2) - 0.5 * ws),
                   (int)((bs / 2) - 0.5 * hs),
                   (int)((bs / 2) + 0.5 * (ws - 1)),
                   (int)((bs / 2) + 0.5 * (hs - 1)));
            DEBUG_PRINTF("%f %f %f %f\n",
                   bs/2, bs/2, ws, hs);
	    */
        }
    }
    return STATUS_SUCCESS;
}

frcnnStatus_t generateAnchors(cudaStream_t stream,
                              int numRatios,
                              float* ratios,
                              int numScales,
                              float* scales,
                              int baseSize,
                              float* anchors)
{
    int ac = numRatios * numScales * 4;
    float* anchors_cpu;
    cudaMallocHost((void**) &anchors_cpu, sizeof(float) * ac);
    frcnnStatus_t status = generateAnchors_cpu(numRatios, ratios, numScales, scales, baseSize, anchors_cpu);
    cudaMemcpyAsync(anchors, anchors_cpu, sizeof(float) * ac, cudaMemcpyHostToDevice, stream);
    cudaFreeHost(anchors_cpu);
    return status;
}
