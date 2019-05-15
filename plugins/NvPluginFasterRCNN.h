#ifndef TRT_NV_PLUGIN_FASTER_RCNN_H
#define TRT_NV_PLUGIN_FASTER_RCNN_H

#include "NvInferPlugin.h"
#include "plugin.h"
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class RPROIPlugin : public BasePlugin
{
public:
    RPROIPlugin(RPROIParams params, const float* anchorsRatios, const float* anchorsScales);

    RPROIPlugin(const void* data, size_t length);

    ~RPROIPlugin() override;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override { return 0; }

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
    float* copyToHost(const void* srcHostData, int count);

    int copyFromHost(char* dstHostBuffer, const void* source, int count) const;

    template <typename VALUE>
    const VALUE* read(const char** ptr)
    {
        const auto* t = reinterpret_cast<const VALUE*>(*ptr);
        (*ptr) += sizeof(VALUE);
        return t;
    }

    // These won't be serialized
    float* anchorsDev;

    // These need to be serialized
    RPROIParams params;
    int A, C, H, W;
    float *anchorsRatiosHost, *anchorsScalesHost;
};

class RPROIPluginCreator : public BaseCreator
{
public:
    RPROIPluginCreator();

    ~RPROIPluginCreator() override;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    RPROIParams params;
    std::vector<float> anchorsRatios;
    std::vector<float> anchorsScales;
    static std::vector<PluginField> mPluginAttributes;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_NV_PLUGIN_FASTER_RCNN_H
