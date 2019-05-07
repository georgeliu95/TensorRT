#include "clipPlugin.h"
#include "NvInfer.h"
#include "checkMacrosPlugin.h"
#include "clip.h"
#include "plugin.h"

#include <cstring>
#include <cudnn.h>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::ClipPluginCreator;
using nvinfer1::plugin::ClipPlugin;

static const char* CLIP_PLUGIN_VERSION{"1"};
static const char* CLIP_PLUGIN_NAME{"Clip_TRT"};
PluginFieldCollection ClipPluginCreator::mFC{};
std::vector<PluginField> ClipPluginCreator::mPluginAttributes;

ClipPlugin::ClipPlugin(std::string name, float clipMin, float clipMax)
    : mLayerName(std::move(name))
    , mClipMin(clipMin)
    , mClipMax(clipMax)
{
}

ClipPlugin::ClipPlugin(std::string name, const void* data, size_t length)
    : mLayerName(std::move(name))
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data), *a = d;

    mClipMin = read<float>(d);
    mClipMax = read<float>(d);
    mDataType = read<DataType>(d);
    mInputVolume = read<size_t>(d);

    ASSERT(d == (a + length));
}

const char* ClipPlugin::getPluginType() const
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPlugin::getPluginVersion() const
{
    return CLIP_PLUGIN_VERSION;
}

int ClipPlugin::getNbOutputs() const
{
    return 1;
}

Dims ClipPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return *inputs;
}

int ClipPlugin::initialize()
{
    return 0;
}

int ClipPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    void* output = outputs[0];
    int status = pluginStatus_t::STATUS_FAILURE;
    status = clipInference(stream, mInputVolume * batchSize, mClipMin, mClipMax, inputs[0], output, mDataType);

    if (status != pluginStatus_t::STATUS_SUCCESS)
    {
        gLogError << "ClipPlugin Kernel failed for layer name " << mLayerName << std::endl;
    }

    return status;
}

size_t ClipPlugin::getSerializationSize() const
{
    return 2 * sizeof(float) + sizeof(mDataType) + sizeof(mInputVolume);
}

void ClipPlugin::serialize(void* buffer) const
{
    char *d = static_cast<char *>(buffer), *a = d;

    //Serialize plugin data
    nvinfer1::plugin::write(d, mClipMin);
    nvinfer1::plugin::write(d, mClipMax);
    nvinfer1::plugin::write(d, mDataType);
    nvinfer1::plugin::write(d, mInputVolume);

    if (d != a + getSerializationSize())
    {
        gLogError << "ClipPlugin serialize failed for layer name " << mLayerName << std::endl;
    }

    ASSERT(d == a + getSerializationSize());
}

void ClipPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    ASSERT(nbOutputs == 1);
    API_CHECK_ENUM_RANGE(DataType, type);
    API_CHECK_ENUM_RANGE(PluginFormat, format);
    mDataType = type;

    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++)
    {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
}

bool ClipPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    if (type == DataType::kINT8)
    {
        return false;
    }

    API_CHECK_ENUM_RANGE_RETVAL(DataType, type, false);
    API_CHECK_ENUM_RANGE_RETVAL(PluginFormat, format, false);
    return true;
}

void ClipPlugin::terminate()
{
}

ClipPlugin::~ClipPlugin() = default;

void ClipPlugin::destroy() { delete this; }

IPluginV2* ClipPlugin::clone() const
{
    ClipPlugin* ret = new ClipPlugin(mLayerName, mClipMin, mClipMax);
    ret->mInputVolume = mInputVolume;
    return ret;
}

ClipPluginCreator::ClipPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("clipMin", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipMax", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ClipPluginCreator::getPluginName() const
{
    return CLIP_PLUGIN_NAME;
}

const char* ClipPluginCreator::getPluginVersion() const
{
    return CLIP_PLUGIN_VERSION;
}

const PluginFieldCollection* ClipPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* ClipPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    float clipMin = 0.0, clipMax = 0.0;
    const PluginField* fields = fc->fields;

    ASSERT(fc->nbFields == 2);
    for (int i = 0; i < fc->nbFields; i++)
    {
        if (strcmp(fields[i].name, "clipMin") == 0)
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            clipMin = *(static_cast<const float*>(fields[i].data));
        }
        else if (strcmp(fields[i].name, "clipMax") == 0)
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            clipMax = *(static_cast<const float*>(fields[i].data));
        }
    }

    return new ClipPlugin(name, clipMin, clipMax);
}

IPluginV2* ClipPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call ClipPlugin::destroy()
    return new ClipPlugin(name, serialData, serialLength);
}
