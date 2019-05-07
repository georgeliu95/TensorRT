#include "reorgPlugin.h"
#include "checkMacrosPlugin.h"
#include "yolo.h"

using namespace nvinfer1;
using nvinfer1::PluginType;
using nvinfer1::plugin::ReorgPluginCreator;
using nvinfer1::plugin::Reorg;

static const char* REORG_PLUGIN_VERSION{"1"};
static const char* REORG_PLUGIN_NAME{"Reorg_TRT"};
PluginFieldCollection ReorgPluginCreator::mFC{};
std::vector<PluginField> ReorgPluginCreator::mPluginAttributes;

Reorg::Reorg(int stride)
    : stride(stride)
{
}

Reorg::Reorg(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    stride = read<int>(d);
    ASSERT(d == a + length);
}

int Reorg::getNbOutputs() const { return 1; }

Dims Reorg::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return DimsCHW(inputs[0].d[0] * stride * stride, inputs[0].d[1] / stride, inputs[0].d[2] / stride);
}

int Reorg::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    yoloStatus_t status = reorgInference(stream,
                                         batchSize, C, H, W, stride,
                                         inputData, outputData);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

size_t Reorg::getSerializationSize() const
{
    // C, H, W, stride
    return sizeof(int) * 4;
}

void Reorg::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, stride);
    ASSERT(d == a + getSerializationSize());
}

void Reorg::configureWithFormat(const Dims* inputDims, int nbInputs,
                                const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int)
{
    ASSERT(type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);
    ASSERT(stride > 0);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    ASSERT(H % stride == 0);
    ASSERT(W % stride == 0);
}

bool Reorg::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

int Reorg::initialize() { return 0; }

void Reorg::terminate() {}

size_t Reorg::getWorkspaceSize(int maxBatchSize) const { return 0; }

const char* Reorg::getPluginType() const { return REORG_PLUGIN_NAME; }

const char* Reorg::getPluginVersion() const { return REORG_PLUGIN_VERSION; }

void Reorg::destroy() { delete this; }

IPluginV2* Reorg::clone() const
{
    IPluginV2* plugin = new Reorg(stride);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

ReorgPluginCreator::ReorgPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ReorgPluginCreator::getPluginName() const
{
    return REORG_PLUGIN_NAME;
}

const char* ReorgPluginCreator::getPluginVersion() const
{
    return REORG_PLUGIN_VERSION;
}

const PluginFieldCollection* ReorgPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* ReorgPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    ASSERT(fc->nbFields == 1);
    ASSERT(fields[0].type == PluginFieldType::kINT32);
    stride = static_cast<int>(*(static_cast<const int*>(fields[0].data)));

    return new Reorg(stride);
}

IPluginV2* ReorgPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    //This object will be deleted when the network is destroyed, which will
    //call ReorgPlugin::destroy()
    return new Reorg(serialData, serialLength);
}
