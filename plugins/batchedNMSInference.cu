#include "ssd.h"
#include "ssdMacros.h"
#include "ssd_internal.h"

namespace nvinfer1
{
namespace plugin
{
ssdStatus_t nmsInference(
    cudaStream_t stream,
    const int N,
    const int perBatchBoxesSize,
    const int perBatchScoresSize,
    const bool shareLocation,
    const int backgroundLabelId,
    const int numPredsPerClass,
    const int numClasses,
    const int topK,
    const int keepTopK,
    const float scoreThreshold,
    const float iouThreshold,
    const DType_t DT_BBOX,
    const void* locData,
    const DType_t DT_SCORE,
    const void* confData,
    void* keepCount,
    void* nmsedBoxes,
    void* nmsedScores,
    void* nmsedClasses,
    void* workspace,
    bool isNormalized,
    bool confSigmoid,
    bool clipBoxes)
{
    const int locCount = N * perBatchBoxesSize;
    const int numLocClasses = shareLocation ? 1 : numClasses;

    size_t bboxDataSize = detectionForwardBBoxDataSize(N, perBatchBoxesSize, DataType::kFLOAT);
    void* bboxDataRaw = workspace;
    cudaMemcpyAsync(bboxDataRaw, locData, bboxDataSize,
        cudaMemcpyDeviceToDevice, stream);
    ssdStatus_t status;

    // float for now
    void* bboxData;
    size_t bboxPermuteSize = detectionForwardBBoxPermuteSize(shareLocation, N, perBatchBoxesSize, DataType::kFLOAT);
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

    const int numScores = N * perBatchScoresSize;
    size_t totalScoresSize = detectionForwardPreNMSSize(N, perBatchScoresSize);
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
    
    size_t indicesSize = detectionForwardPreNMSSize(N, perBatchScoresSize);
    void* indices = nextWorkspacePtr((int8_t*) scores, totalScoresSize);

    size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topK);
    size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topK);
    void* postNMSScores = nextWorkspacePtr((int8_t*) indices, indicesSize);
    void* postNMSIndices = nextWorkspacePtr((int8_t*) postNMSScores, postNMSScoresSize);

    void* sortingWorkspace = nextWorkspacePtr((int8_t*) postNMSIndices, postNMSIndicesSize);
    status = sortScoresPerClass(stream,
                                N,
                                numClasses,
                                numPredsPerClass,
                                backgroundLabelId,
                                scoreThreshold,
                                DataType::kFLOAT,
                                scores,
                                indices,
                                sortingWorkspace);

    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    //This is set to true as the input bounding boxes are of the format [ymin,
    //xmin, ymax, xmax]. The default implementation assumes [xmin, ymin, xmax, ymax] 
    bool flipXY = true; 
    status = allClassNMS(stream,
                         N,
                         numClasses,
                         numPredsPerClass,
                         topK,
                         iouThreshold,
                         shareLocation,
                         isNormalized,
                         DataType::kFLOAT,
                         DataType::kFLOAT,
                         bboxData,
                         scores,
                         indices,
                         postNMSScores,
                         postNMSIndices, 
                         flipXY);
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

    status = gatherNMSOutputs(stream,
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
                              nmsedBoxes,
                              nmsedScores,
                              nmsedClasses,
                              clipBoxes);
    SSD_ASSERT_FAILURE(status == STATUS_SUCCESS);

    return STATUS_SUCCESS;
}

} // namespace plugin
} // namespace nvinfer1
