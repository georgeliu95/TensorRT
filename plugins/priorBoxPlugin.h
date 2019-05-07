#ifndef TRT_PRIOR_BOX_PLUGIN_H
#define TRT_PRIOR_BOX_PLUGIN_H
#include "NvInferPlugin.h"
#include "cudnn.h"
#include "plugin.h"
#include <cassert>
#include <cublas_v2.h>
#include <cstdlib>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class PriorBox : public BasePlugin
{
public:
    PriorBox(PriorBoxParameters param);

    PriorBox(const void* buffer, size_t length);

    ~PriorBox() override= default;;

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

    PriorBoxParameters mParam{};
    int numPriors{};
    int H{};
    int W{};
    Weights minSize{};
    Weights maxSize{};
    Weights aspectRatios{}; // not learnable weights
};

class PriorBoxPluginCreator : public BaseCreator
{
public:
    PriorBoxPluginCreator();

    ~PriorBoxPluginCreator() override;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    template <typename T>
    T* allocMemory(int size = 1)
    {
        mTmpAllocs.reserve(mTmpAllocs.size() + 1);
        T* tmpMem = static_cast<T*>(malloc(sizeof(T) * size));
        mTmpAllocs.push_back(tmpMem);
        return tmpMem;
    }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::vector<void*> mTmpAllocs;
};
}
}

#endif // TRT_PRIOR_BOX_PLUGIN_H
