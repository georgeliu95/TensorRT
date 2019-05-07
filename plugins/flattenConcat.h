#ifndef TRT_FLATTENCONCAT_PLUGIN_H
#define TRT_FLATTENCONCAT_PLUGIN_H

#include "NvInferPlugin.h"
#include "plugin.h"
#include <cstdlib>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

#define LOG_ERROR(status)                                      \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

namespace nvinfer1
{
namespace plugin
{
class FlattenConcat : public BasePlugin
{
public:
    FlattenConcat(int concatAxis, bool ignoreBatch);

    FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis, const int* inputConcatAxis);

    FlattenConcat(const void* data, size_t length);

    ~FlattenConcat() override;

    FlattenConcat() = delete;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int) const override;

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
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    size_t* mCopySize{};
    bool mIgnoreBatch{};
    int mConcatAxisID{};
    int mOutputConcatAxis{};
    int mNumInputs{};
    int* mInputConcatAxis{};
    nvinfer1::Dims mCHW{0, {}, {}};
    cublasHandle_t mCublas{};
};

class FlattenConcatPluginCreator : public BaseCreator
{
public:
    FlattenConcatPluginCreator();

    ~FlattenConcatPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    bool mIgnoreBatch{};
    int mConcatAxisID{};
    static std::vector<PluginField> mPluginAttributes;
};

} //namesplace plugin
} //namespace nvinfer1

#endif // TRT_FLATTENCONCAT_PLUGIN_H
