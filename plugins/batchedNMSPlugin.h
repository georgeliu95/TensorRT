#ifndef TRT_BATCHED_NMS_PLUGIN_H
#define TRT_BATCHED_NMS_PLUGIN_H
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

class BatchedNMSPlugin : public IPluginV2Ext
{
public:
    BatchedNMSPlugin(NMSParameters param);

    BatchedNMSPlugin(const void* data, size_t length);

    ~BatchedNMSPlugin() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                         const DataType* inputTypes, const DataType* outputTypes,
                         const bool* inputIsBroadcast, const bool* outputIsBroadcast,
                         PluginFormat floatFormat, int maxBatchSize) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType, int nbInputs) const override;

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void setClipParam(bool clip) { mClipBoxes = clip; }
private:
    NMSParameters param{};
    int boxesSize{};
    int scoresSize{};
    int numPriors{};
    std::string mNamespace;
    bool mClipBoxes{};
};

class BatchedNMSPluginCreator : public BaseCreator
{
public:
    BatchedNMSPluginCreator();

    ~BatchedNMSPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    NMSParameters params{};
    static std::vector<PluginField> mPluginAttributes;
    bool mClipBoxes{};
};
} //namespace plugin
} //namespace nvinfer1

#endif // TRT_BATCHED_NMS_PLUGIN_H
