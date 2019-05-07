#ifndef TRT_SSD_H
#define TRT_SSD_H
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cublas_v2.h"
#include "plugin.h"
#include <cstddef>
#include <string>
#include <vector>

using nvinfer1::plugin::CodeTypeSSD;
using nvinfer1::plugin::PriorBoxParameters;
using nvinfer1::plugin::GridAnchorParameters;

typedef nvinfer1::DataType DType_t;
using nvinfer1::DataType;

typedef enum {
    WARP = 1,
    FIT_SMALL_SIZE = 2,
    FIT_LARGE_SIZE_AND_PAD = 3
} ResizeMode;

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

namespace nvinfer1
{
namespace plugin
{

struct ResizeParameters
{
    ResizeMode mode;
    int height, width, heightScale, widthScale;
};

// new score/index array based NMS (avoid memory move of large volume)
ssdStatus_t allClassNMS(cudaStream_t stream,
                        int num,
                        int num_classes,
                        int num_preds_per_class,
                        int top_k,
                        float nms_threshold,
                        bool share_location,
                        bool isNormalized,
                        DType_t DT_SCORE,
                        DType_t DT_BBOX,
                        void* bbox_data,
                        void* beforeNMS_scores,
                        void* beforeNMS_index_array,
                        void* afterNMS_scores,
                        void* afterNMS_index_array,
                        bool flipXY=false);

ssdStatus_t getSortedBboxInfo(cudaStream_t stream,
                              int num,
                              int num_classes,
                              int top_k,
                              DType_t DT_CONF,
                              void* bboxinfo_gpu,
                              void* temp_scores_gpu,
                              void* sorted_bboxinfo_gpu,
                              void* workspace);

size_t sortedBboxInfoWorkspaceSize(
    int num,
    int num_classes,
    int top_k,
    DType_t DT_CONF);

ssdStatus_t sortScoresPerClass(
    cudaStream_t stream,
    int num,
    int num_classes,
    int num_preds_per_class,
    int background_label_id,
    float confidence_threshold,
    DType_t DT_SCORE,
    void* conf_scores_gpu,
    void* index_array_gpu,
    void* workspace);

size_t sortScoresPerClassWorkspaceSize(
    int num,
    int num_classes,
    int num_preds_per_class,
    DType_t DT_CONF);

ssdStatus_t sortScoresPerImage(
    cudaStream_t stream,
    int num_images,
    int num_items_per_image,
    DType_t DT_SCORE,
    void* unsorted_scores,
    void* unsorted_bbox_indices,
    void* sorted_scores,
    void* sorted_bbox_indices,
    void* workspace);

size_t sortScoresPerImageWorkspaceSize(
    int num_images,
    int num_items_per_image,
    DType_t DT_SCORE);

ssdStatus_t permuteData(
    cudaStream_t stream,
    int nthreads,
    int num_classes,
    int num_data,
    int num_dim,
    DType_t DT_DATA,
    bool confSigmoid,
    const void* data,
    void* new_data);

ssdStatus_t decodeBBoxes(
    cudaStream_t stream,
    int nthreads,
    CodeTypeSSD code_type,
    bool variance_encoded_in_target,
    int num_priors,
    bool share_location,
    int num_loc_classes,
    int background_label_id,
    bool clip_bbox,
    DType_t DT_BBOX,
    const void* loc_data,
    const void* prior_data,
    void* bbox_data);

ssdStatus_t gatherTopDetections(
    cudaStream_t stream,
    bool shareLocation,
    int numImages,
    int numPredsPerClass,
    int numClasses,
    int topK,
    int keepTopK,
    DType_t DT_BBOX,
    DType_t DT_SCORE,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* keepCount,
    void* topDetections);

ssdStatus_t detectionInference(
    cudaStream_t stream,
    int N,
    int C1,
    int C2,
    bool shareLocation,
    bool varianceEncodedInTarget,
    int backgroundLabelId,
    int numPredsPerClass,
    int numClasses,
    int topK,
    int keepTopK,
    float confidenceThreshold,
    float nmsThreshold,
    CodeTypeSSD codeType,
    DType_t DT_BBOX,
    const void* locData,
    const void* priorData,
    DType_t DT_SCORE,
    const void* confData,
    void* keepCount,
    void* topDetections,
    void* workspace,
    bool isNormalized = true,
    bool confSigmoid = false);

size_t detectionForwardBBoxDataSize(int N,
                                    int C1,
                                    DType_t DT_BBOX);

size_t detectionForwardBBoxPermuteSize(bool shareLocation,
                                       int N,
                                       int C1,
                                       DType_t DT_BBOX);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPostNMSSize(int N,
                                   int numClasses,
                                   int topK);

size_t detectionInferenceWorkspaceSize(bool shareLocation,
                                       int N,
                                       int C1,
                                       int C2,
                                       int numClasses,
                                       int numPredsPerClass,
                                       int topK,
                                       DType_t DT_BBOX,
                                       DType_t DT_SCORE);

ssdStatus_t normalizeInference(
    cudaStream_t stream,
    cublasHandle_t handle,
    bool acrossSpatial,
    bool channelShared,
    int N,
    int C,
    int H,
    int W,
    float eps,
    const void* scale,
    const void* inputData,
    void* outputData,
    void* workspace);

size_t normalizePluginWorkspaceSize(bool acrossSpatial, int C, int H, int W);

ssdStatus_t permuteInference(
    cudaStream_t stream,
    bool needPermute,
    int numAxes,
    int count,
    const void* permuteOrder,
    const void* oldSteps,
    const void* newSteps,
    const void* inputData,
    void* outputData);

ssdStatus_t priorBoxInference(
    cudaStream_t stream,
    PriorBoxParameters param,
    int H,
    int W,
    int numPriors,
    int numAspectRatios,
    const void* minSize,
    const void* maxSize,
    const void* aspectRatios,
    void* outputData);

ssdStatus_t anchorGridInference(
    cudaStream_t stream,
    GridAnchorParameters param,
    int numAspectRatios,
    const void* aspectRatios,
    const void* scales,
    void* outputData);

ssdStatus_t nmsInference(
    cudaStream_t stream,
    int N,
    int boxesSize,
    int scoresSize,
    bool shareLocation,
    int backgroundLabelId,
    int numPredsPerClass, 
    int numClasses,
    int topK,
    int keepTopK,
    float scoreThreshold,
    float iouThreshold,
    DType_t DT_BBOX,
    const void* locData,
    DType_t DT_SCORE,
    const void* confData,
    void* keepCount,
    void* nmsedBoxes,
    void* nmsedScores,
    void* nmsedClasses,
    void* workspace,
    bool isNormalized=true,
    bool confSigmoid=false,
    bool clipBoxes=true);

ssdStatus_t gatherNMSOutputs(
    cudaStream_t stream,
    bool shareLocation,
    int numImages,
    int numPredsPerClass,
    int numClasses,
    int topK,
    int keepTopK,
    DType_t DT_BBOX,
    DType_t DT_SCORE,
    const void* indices,
    const void* scores,
    const void* bboxData,
    void* keepCount,
    void* nmsedBoxes, 
    void* nmsedScores, 
    void* nmsedClasses,
    bool clipBoxes=true);


} // namespace plugin
} // namespace nvinfer1

#endif // TRT_SSD_H