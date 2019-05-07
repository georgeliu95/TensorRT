#include "nmsPlugin.h"
#include "checkMacrosPlugin.h"
#include "ssd.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::NMSPluginCreator;
using nvinfer1::plugin::DetectionOutput;
using nvinfer1::plugin::DetectionOutputParameters;

namespace
{
const char* NMS_PLUGIN_VERSION{"1"};
const char* NMS_PLUGIN_NAME{"NMS_TRT"};
}

PluginFieldCollection NMSPluginCreator::mFC{};
std::vector<PluginField> NMSPluginCreator::mPluginAttributes;

DetectionOutput::DetectionOutput(DetectionOutputParameters params)
    : param(params)
{
}

DetectionOutput::DetectionOutput(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    param = read<DetectionOutputParameters>(d);
    C1 = read<int>(d);
    C2 = read<int>(d);
    numPriors = read<int>(d);
    assert(d == a + length);
}

int DetectionOutput::getNbOutputs() const
{
    return 2;
}

int DetectionOutput::initialize()
{
    return 0;
}

void DetectionOutput::terminate()
{
}

Dims DetectionOutput::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 3);
    assert(index == 0 || index == 1);
    C1 = inputs[param.inputOrder[0]].d[0], C2 = inputs[param.inputOrder[1]].d[0];
    if (index == 0)
    {
        return DimsCHW(1, param.keepTopK, 7);
    }
#if 0 // FIXME: why is this code here?
    return DimsCHW(1, param.keepTopK, 1);
#else
    return DimsCHW(1, 1, 1);
#endif
}

size_t DetectionOutput::getWorkspaceSize(int maxBatchSize) const
{
    return detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, C1, C2, param.numClasses, numPriors, param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

int DetectionOutput::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* const locData = inputs[param.inputOrder[0]];
    const void* const confData = inputs[param.inputOrder[1]];
    const void* const priorData = inputs[param.inputOrder[2]];

    void* topDetections = outputs[0];
    void* keepCount = outputs[1];

    ssdStatus_t status = detectionInference(stream,
                                            batchSize,
                                            C1,
                                            C2,
                                            param.shareLocation,
                                            param.varianceEncodedInTarget,
                                            param.backgroundLabelId,
                                            numPriors,
                                            param.numClasses,
                                            param.topK,
                                            param.keepTopK,
                                            param.confidenceThreshold,
                                            param.nmsThreshold,
                                            param.codeType,
                                            DataType::kFLOAT,
                                            locData,
                                            priorData,
                                            DataType::kFLOAT,
                                            confData,
                                            keepCount,
                                            topDetections,
                                            workspace,
                                            param.isNormalized,
                                            param.confSigmoid);
    assert(status == STATUS_SUCCESS);
    return 0;
}

size_t DetectionOutput::getSerializationSize() const
{
    // DetectionOutputParameters, C1,C2,numPriors
    return sizeof(DetectionOutputParameters) + sizeof(int) * 3;
}

void DetectionOutput::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, param);
    write(d, C1);
    write(d, C2);
    write(d, numPriors);
    assert(d == a + getSerializationSize());
}

void DetectionOutput::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
{
    assert(nbInputs == 3);
    assert(nbOutputs == 2);
    assert(inputDims[0].nbDims == 3);
    assert(inputDims[1].nbDims == 3);
    assert(inputDims[2].nbDims == 3);
    assert(outputDims[0].nbDims == 3);
    assert(outputDims[1].nbDims == 3);
    C1 = inputDims[param.inputOrder[0]].d[0];
    C2 = inputDims[param.inputOrder[1]].d[0];
    numPriors = inputDims[param.inputOrder[2]].d[1] / 4;
    const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
    assert(numPriors * numLocClasses * 4 == inputDims[param.inputOrder[0]].d[0]);
    assert(numPriors * param.numClasses == inputDims[param.inputOrder[1]].d[0]);
}

bool DetectionOutput::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}
const char* DetectionOutput::getPluginType() const { return NMS_PLUGIN_NAME; }

const char* DetectionOutput::getPluginVersion() const { return NMS_PLUGIN_VERSION; }

void DetectionOutput::destroy() { delete this; }

IPluginV2* DetectionOutput::clone() const
{
    IPluginV2* plugin = new DetectionOutput(param);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

NMSPluginCreator::NMSPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("varianceEncodedInTarget", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("confidenceThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("nmsThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("inputOrder", nullptr, PluginFieldType::kINT32, 3));
    mPluginAttributes.emplace_back(PluginField("confSigmoid", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("codeType", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* NMSPluginCreator::getPluginName() const
{
    return NMS_PLUGIN_NAME;
}

const char* NMSPluginCreator::getPluginVersion() const
{
    return NMS_PLUGIN_VERSION;
}

const PluginFieldCollection* NMSPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* NMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    //Default init values for TF SSD network
    params.codeType = CodeTypeSSD::TF_CENTER;
    params.inputOrder[0] = 0;
    params.inputOrder[1] = 2;
    params.inputOrder[2] = 1;

    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "shareLocation"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.shareLocation = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "varianceEncodedInTarget"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.varianceEncodedInTarget = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "backgroundLabelId"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.backgroundLabelId = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "numClasses"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.numClasses = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "topK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.topK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "keepTopK"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.keepTopK = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confidenceThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.confidenceThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "nmsThreshold"))
        {
            ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            params.nmsThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "confSigmoid"))
        {
            params.confSigmoid = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "isNormalized"))
        {
            params.isNormalized = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "inputOrder"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            int size = fields[i].length;
            const int* o = static_cast<const int*>(fields[i].data);
            for (int j = 0; j < size; j++)
            {
                params.inputOrder[j] = *o;
                o++;
            }
        }
        else if (!strcmp(attrName, "codeType"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.codeType = static_cast<CodeTypeSSD>(*(static_cast<const int*>(fields[i].data)));
        }
    }

    return new DetectionOutput(params);
}

IPluginV2* NMSPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    //This object will be deleted when the network is destroyed, which will
    //call NMS::destroy()
    return new DetectionOutput(serialData, serialLength);
}
