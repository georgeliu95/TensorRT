#ifndef TRT_CLIP_PLUGIN_H
#define TRT_CLIP_PLUGIN_H

#include "NvInferPlugin.h"
#include "plugin.h"
#include <cstdlib>
#include <cudnn.h>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{
class ClipPlugin : public BasePlugin
{
public:
    ClipPlugin(std::string name, float clipMin, float clipMax);

    ClipPlugin(std::string name, const void* data, size_t length);

    ~ClipPlugin() override;

    ClipPlugin() = delete;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int) const override { return 0; };

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
    std::string mLayerName;
    float mClipMin{0.0f};
    float mClipMax{0.0f};
    DataType mDataType{DataType::kFLOAT};
    size_t mInputVolume{0};
};

class ClipPluginCreator : public BaseCreator
{
public:
    ClipPluginCreator();

    ~ClipPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

} //namesplace plugin
} //namespace nvinfer1

#endif // TRT_CLIP_PLUGIN_H
