#ifndef TRT_NORMALIZE_PLUGIN_H
#define TRT_NORMALIZE_PLUGIN_H
#include "NvInferPlugin.h"
#include "cudnn.h"
#include "plugin.h"
#include <cassert>
#include <cublas_v2.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class Normalize : public BasePlugin
{
public:
    Normalize(const Weights* weights, int nbWeights, bool acrossSpatial, bool channelShared, float eps);

    Normalize(const void* buffer, size_t length);

    ~Normalize() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2* clone() const override;

private:
    Weights copyToDevice(const void* hostData, size_t count);
    void serializeFromDevice(char*& hostBuffer, Weights deviceWeights) const;
    Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    cublasHandle_t mCublas{};

    int C{};
    int H{};
    int W{};
    int mNbWeights{};
    bool acrossSpatial{};
    bool channelShared{};
    float eps{};
    Weights mWeights{};
};

class NormalizePluginCreator : public BaseCreator
{
public:
    NormalizePluginCreator();

    ~NormalizePluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    bool mAcrossSpatial{};
    bool mChannelShared{};
    float mEps{};
    int mNbWeights{};
    static std::vector<PluginField> mPluginAttributes;
};
}
}

#endif // TRT_NORMALIZE_PLUGIN_H
