#include "ssd.h"
#include "ssdMacros.h"
#include "ssd_internal.h"

namespace nvinfer1
{
namespace plugin
{
ssdStatus_t detectionInference(
    cudaStream_t stream,
    const int N,
    const int C1,
    const int C2,
    const bool shareLocation,
    const bool varianceEncodedInTarget,
    const int backgroundLabelId,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const float confidenceThreshold,
    const float nmsThreshold,
    const CodeTypeSSD codeType,
    const DType_t DT_BBOX,
    const void* locData,
    const void* priorData,
    const DType_t DT_SCORE,
    const void* confData,
    void* keepCount,
    void* topDetections,
    void* workspace,
    bool isNormalized,
    bool confSigmoid)
{
    const int locCount = N * C1;
    const bool clipBBox = false;
    const int numLocClasses = shareLocation ? 1 : numClasses;

    size_t bboxDataSize = detectionForwardBBoxDataSize(N, C1, DataType::kFLOAT);
    void* bboxDataRaw = workspace;

    ssdStatus_t status = decodeBBoxes(stream,
                                      locCount,
                                      codeType,
                                      varianceEncodedInTarget,
                                      numPredsPerClass,
                                      shareLocation,
                                      numLocClasses,
                                      backgroundLabelId,
                                      clipBBox,
                                      DataType::kFLOAT,
                                      locData,
                                      priorData,
                                      bboxDataRaw);

    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    // float for now
    void* bboxData;
    size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DataType::kFLOAT);
    void* bboxPermute = nextWorkspacePtr((int8_t*) bboxDataRaw, bboxDataSize);

    if (!shareLocation)
    {
        status = permuteData(stream,
                             locCount,
                             numLocClasses,
                             numPredsPerClass,
                             4,
                             DataType::kFLOAT,
                             false,
                             bboxDataRaw,
                             bboxPermute);
        SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);
        bboxData = bboxPermute;
    }
    else
    {
        bboxData = bboxDataRaw;
    }

    const int numScores = N * C2;
    size_t scoresSize = detectionForwardPreNMSSize(N, C2);
    void* scores = nextWorkspacePtr((int8_t*) bboxPermute, bboxPermuteSize);
    // need a conf_scores
    status = permuteData(stream,
                         numScores,
                         numClasses,
                         numPredsPerClass,
                         1,
                         DataType::kFLOAT,
                         confSigmoid,
                         confData,
                         scores);
    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    size_t indicesSize = detectionForwardPreNMSSize(N, C2);
    void* indices = nextWorkspacePtr((int8_t*) scores, scoresSize);

    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void* postNMSScores = nextWorkspacePtr((int8_t*) indices, indicesSize);
    void* postNMSIndices = nextWorkspacePtr((int8_t*) postNMSScores, postNMSScoresSize);

    //size_t sortingWorkspaceSize = sortScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, FLOAT32);
    void* sortingWorkspace = nextWorkspacePtr((int8_t*) postNMSIndices, postNMSIndicesSize);
    status = sortScoresPerClass(stream,
                                N,
                                numClasses,
                                numPredsPerClass,
                                backgroundLabelId,
                                confidenceThreshold,
                                DataType::kFLOAT,
                                scores,
                                indices,
                                sortingWorkspace);
    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    status = allClassNMS(stream,
                         N,
                         numClasses,
                         numPredsPerClass,
                         topK,
                         nmsThreshold,
                         shareLocation,
                         isNormalized,
                         DataType::kFLOAT,
                         DataType::kFLOAT,
                         bboxData,
                         scores,
                         indices,
                         postNMSScores,
                         postNMSIndices,
                         false);
    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    status = sortScoresPerImage(stream,
                                N,
                                numClasses * topK,
                                DataType::kFLOAT,
                                postNMSScores,
                                postNMSIndices,
                                scores,
                                indices,
                                sortingWorkspace);
    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    status = gatherTopDetections(stream,
                                 shareLocation,
                                 N,
                                 numPredsPerClass,
                                 numClasses,
                                 topK,
                                 keepTopK,
                                 DataType::kFLOAT,
                                 DataType::kFLOAT,
                                 indices,
                                 scores,
                                 bboxData,
                                 keepCount,
                                 topDetections);
    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

} // namespace plugin
} // namespace nvinfer1
