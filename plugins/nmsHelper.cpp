#include "ssd.h"
#include "ssdMacros.h"
#include <cassert>
#include <algorithm>

namespace nvinfer1
{
namespace plugin
{

size_t detectionForwardBBoxDataSize(int N,
                                    int C1,
                                    DType_t DT_BBOX)
{
    if (DT_BBOX == DataType::kFLOAT)
    {
        return N * C1 * sizeof(float);
    }
    
    printf("Only FP32 type bounding boxes are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardBBoxPermuteSize(bool shareLocation,
                                       int N,
                                       int C1,
                                       DType_t DT_BBOX)
{
    if (DT_BBOX == DataType::kFLOAT)
    {
        return shareLocation ? 0 : N * C1 * sizeof(float);
    }
    printf("Only FP32 type bounding boxes are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardPreNMSSize(int N,
                                  int C2)
{
    static_assert(sizeof(float) == sizeof(int), "Must run on a platform where sizeof(int) == sizeof(float)");
    return N * C2 * sizeof(float);
}

size_t detectionForwardPostNMSSize(int N,
                                   int numClasses,
                                   int topK)
{
    static_assert(sizeof(float) == sizeof(int), "Must run on a platform where sizeof(int) == sizeof(float)");
    return N * numClasses * topK * sizeof(float);
}

size_t detectionInferenceWorkspaceSize(bool shareLocation,
                                       int N,
                                       int C1,
                                       int C2,
                                       int numClasses,
                                       int numPredsPerClass,
                                       int topK,
                                       DType_t DT_BBOX,
                                       DType_t DT_SCORE)
{
    size_t wss[7];
    wss[0] = detectionForwardBBoxDataSize(N, C1, DT_BBOX);
    wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX);
    wss[2] = detectionForwardPreNMSSize(N, C2);
    wss[3] = detectionForwardPreNMSSize(N, C2);
    wss[4] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[5] = detectionForwardPostNMSSize(N, numClasses, topK);
    wss[6] = std::max(sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, DT_SCORE),
                      sortScoresPerImageWorkspaceSize(N, numClasses * topK, DT_SCORE));
    return calculateTotalWorkspaceSize(wss, 7);
}
} //namespace plugin
} //namespace nvinfer1
