#ifndef TRT_NMS_PLUGIN_H
#define TRT_NMS_PLUGIN_H
#include "NvInferPlugin.h"
#include "plugin.h"
#include <cassert>
#include <string>
#include <vector>

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class DetectionOutput : public BasePlugin
{
public:
    DetectionOutput(DetectionOutputParameters param);

    DetectionOutput(const void* data, size_t length);

    ~DetectionOutput() override = default;

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
    DetectionOutputParameters param{};
    int C1{};
    int C2{};
    int numPriors{};
};

class NMSPluginCreator : public BaseCreator
{
public:
    NMSPluginCreator();

    ~NMSPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    DetectionOutputParameters params{};
    static std::vector<PluginField> mPluginAttributes;
};
}
}

#endif // TRT_NMS_PLUGIN_H
