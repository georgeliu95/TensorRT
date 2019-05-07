#ifndef TRT_NV_PLUGINS_LEGACY_H
#define TRT_NV_PLUGINS_LEGACY_H

#include "NvInferPlugin.h"
#include "NvInferPlugin.h"
#include "cudnn.h"
#include "ssd.h"
#include <cublas_v2.h>

namespace nvinfer1
{
namespace plugin
{

struct RPROIParamsLegacy
{
    int poolingH;
    int poolingW;
    int featureStride;
    int preNmsTop;
    int nmsMaxOut;
    int anchorsRatioCount;
    int anchorsScaleCount;
    float iouThreshold;
    float minBoxSize;
    float spatialScale;
};

class RPROIPluginLegacy : public INvPlugin
{
public:
    RPROIPluginLegacy(RPROIParamsLegacy params, const float* anchorsRatios, const float* anchorsScales);

    RPROIPluginLegacy(const void* data, size_t length);

    ~RPROIPluginLegacy() override;

    // IPlugin

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override { return 0; }

    void terminate() override {}

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    float* copyToHost(const void* srcHostData, int count);

    int copyFromHost(char* dstHostBuffer, const void* source, int count);

    template <typename VALUE>
    const VALUE* read(const char** ptr)
    {
        const auto* t = reinterpret_cast<const VALUE*>(*ptr);
        (*ptr) += sizeof(VALUE);
        return t;
    }

    // These won't be serialized
    float* anchorsDev{};

    // These need to be serialized
    RPROIParamsLegacy params{};
    int A{};
    int C{};
    int H{};
    int W{};
    float *anchorsRatiosHost{};
    float *anchorsScalesHost{};
};

// NormalizeLegacy {{{
class NormalizeLegacy : public INvPlugin
{
public:
    NormalizeLegacy(const Weights* weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps);

    NormalizeLegacy(const void* buffer, size_t length);

    ~NormalizeLegacy() override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    Weights copyToDevice(const void* hostData, size_t count);
    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights);
    Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    cublasHandle_t mCublas{};

    int C{};
    int H{};
    int W{};
    bool acrossSpatial{};
    bool channelShared{};
    float eps{};
    Weights mWeights{};
};
// NormalizeLegacy }}}

// PermuteLegacy {{{
class PermuteLegacy : public INvPlugin
{
public:
    PermuteLegacy(Quadruple permuteOrder)
        : permuteOrder(permuteOrder)
    {
    }

    PermuteLegacy(const void* buffer, size_t length);

    ~PermuteLegacy() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    bool needPermute{};
    Quadruple permuteOrder{};
    Quadruple oldSteps{};
    Quadruple newSteps{};
};
// PermuteLegacy }}}

// PriorBoxLegacy {{{
// TODO: this layer do not really use the data of the input tensors
// It only uses their dimensions
// So there's room for optimization by exploiting this removable dependency
class PriorBoxLegacy : public INvPlugin
{
public:
    PriorBoxLegacy(PriorBoxParameters param);

    PriorBoxLegacy(const void* data, size_t length);

    ~PriorBoxLegacy() override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    Weights copyToDevice(const void* hostData, size_t count);

    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights);

    Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    PriorBoxParameters param{};
    int numPriors{};
    int H{};
    int W{};
    Weights minSize{};
    Weights maxSize{};
    Weights aspectRatios{}; // not learnable weights
};
// PriorBoxLegacy }}}

class GridAnchorGeneratorLegacy : public INvPlugin
{
public:
    GridAnchorGeneratorLegacy(GridAnchorParameters* param, int numLayers);

    GridAnchorGeneratorLegacy(const void* data, size_t length);

    ~GridAnchorGeneratorLegacy() override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    Weights copyToDevice(const void* hostData, size_t count);

    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights);

    Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    int mNumLayers;
    std::vector<GridAnchorParameters> mParam;
    int* mNumPriors{};
    Weights *mDeviceWidths{};
    Weights *mDeviceHeights{};
};

// DetectOutput {{{
class DetectionOutputLegacy : public INvPlugin
{
public:
    DetectionOutputLegacy(DetectionOutputParameters param)
        : param(param)
    {
    }

    DetectionOutputLegacy(const void* data, size_t length);

    ~DetectionOutputLegacy() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    DetectionOutputParameters param{};
    int C1{};
    int C2{};
    int numPriors{};
};
// DetectOutput }}}

// ConcatLegacy {{{
class ConcatLegacy : public INvPlugin
{
public:
    ConcatLegacy(int concatAxis, bool ignoreBatch);

    ConcatLegacy(const void* data, size_t length);

    ~ConcatLegacy() override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, int) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    bool mIgnoreBatch{};
    int mConcatAxisID{};
    int mOutputConcatAxis{};
    int mNumInputs{};
    int* mInputConcatAxis{};
    nvinfer1::Dims mCHW{};
    cublasHandle_t mCublas{};
};

// ConcatLegacy }}}
class PReLULegacy : public INvPlugin
{
public:
    PReLULegacy(float negSlope);

    PReLULegacy(const void* buffer, size_t length);

    ~PReLULegacy() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    float mNegSlope{};
    int mBatchDim{};
};

class ReorgLegacy : public INvPlugin
{
public:
    ReorgLegacy(int stride);

    ReorgLegacy(const void* buffer, size_t length);

    ~ReorgLegacy() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    int C{};
    int H{};
    int W{};
    int stride{};
};

class RegionLegacy : public INvPlugin
{
public:
    //RegionLegacy(int num, int coords, int classes, softmaxTree * smTree);
    RegionLegacy(RegionParameters params);

    RegionLegacy(const void* buffer, size_t length);

    ~RegionLegacy() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    int num{};
    int coords{};
    int classes{};
    softmaxTree* smTree{};
    bool hasSoftmaxTree{};
    int C{};
    int H{};
    int W{};
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_NV_PLUGINS_LEGACY_H
