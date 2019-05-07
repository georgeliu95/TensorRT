#include "NvInfer.h"
#include "NvPluginFasterRCNN.h"
#include "checkMacrosPlugin.h"
#include "rpnMacros.h"
#include "rpnlayer.h"
#include <cassert>
#include <cstdio>
#include <iostream>

using namespace nvinfer1;
using nvinfer1::plugin::RPROIPluginCreator;
using nvinfer1::plugin::RPROIPlugin;
using nvinfer1::Dims;
using nvinfer1::PluginType;

namespace
{
const char* RPROI_PLUGIN_VERSION{"1"};
const char* RPROI_PLUGIN_NAME{"RPROI_TRT"};
}

PluginFieldCollection RPROIPluginCreator::mFC{};
std::vector<PluginField> RPROIPluginCreator::mPluginAttributes;

RPROIPlugin::RPROIPlugin(RPROIParams params, const float* anchorsRatios,
                         const float* anchorsScales)
    : params(params)
{
    assert(params.anchorsRatioCount > 0 && params.anchorsScaleCount > 0);
    anchorsRatiosHost = copyToHost(anchorsRatios, params.anchorsRatioCount);
    anchorsScalesHost = copyToHost(anchorsScales, params.anchorsScaleCount);

    CHECK(cudaMalloc((void**) &anchorsDev, 4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float)));
    frcnnStatus_t status = generateAnchors(0,
                                           params.anchorsRatioCount,
                                           anchorsRatiosHost,
                                           params.anchorsScaleCount,
                                           anchorsScalesHost,
                                           params.featureStride,
                                           anchorsDev);
    assert(status == STATUS_SUCCESS);
}

RPROIPlugin::RPROIPlugin(const void* data, size_t length)
    : anchorsDev(nullptr)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    params = *reinterpret_cast<const RPROIParams*>(d);
    d += sizeof(RPROIParams);
    A = *read<int>(&d);
    C = *read<int>(&d);
    H = *read<int>(&d);
    W = *read<int>(&d);
    anchorsRatiosHost = copyToHost(d, params.anchorsRatioCount);
    d += params.anchorsRatioCount * sizeof(float);
    anchorsScalesHost = copyToHost(d, params.anchorsScaleCount);
    d += params.anchorsScaleCount * sizeof(float);
    assert(d == a + length);

    CHECK(cudaMalloc((void**) &anchorsDev, 4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float)));
    frcnnStatus_t status = generateAnchors(0,
                                           params.anchorsRatioCount,
                                           anchorsRatiosHost,
                                           params.anchorsScaleCount,
                                           anchorsScalesHost,
                                           params.featureStride,
                                           anchorsDev);
    assert(status == STATUS_SUCCESS);
}

RPROIPlugin::~RPROIPlugin()
{
    if (anchorsDev != nullptr)
        CHECK(cudaFree(anchorsDev));
    if (anchorsRatiosHost != nullptr)
        CHECK(cudaFreeHost(anchorsRatiosHost));
    if (anchorsScalesHost != nullptr)
        CHECK(cudaFreeHost(anchorsScalesHost));
}

int RPROIPlugin::getNbOutputs() const
{
    return 2;
}

Dims RPROIPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index >= 0 && index < 2);
    assert(nbInputDims == 4);
    assert(inputs[0].nbDims == 3
           && inputs[1].nbDims == 3
           && inputs[2].nbDims == 3
           && inputs[3].nbDims == 3);
    if (index == 0) // rois
    {
        return DimsCHW(1, params.nmsMaxOut, 4);
    }
    else // pool5
    {
        return DimsNCHW(params.nmsMaxOut, inputs[2].d[0], params.poolingH, params.poolingW);
    }
}

size_t RPROIPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return RPROIInferenceFusedWorkspaceSize(maxBatchSize, A, H, W, params.nmsMaxOut);
}

int RPROIPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* const scores = inputs[0];
    const void* const deltas = inputs[1];
    const void* const fmap = inputs[2];
    const void* const iinfo = inputs[3];

    void* rois = outputs[0];
    void* pfmap = outputs[1];

    frcnnStatus_t status = RPROIInferenceFused(stream,
                                               batchSize,
                                               A,
                                               C,
                                               H,
                                               W,
                                               params.poolingH,
                                               params.poolingW,
                                               params.featureStride,
                                               params.preNmsTop,
                                               params.nmsMaxOut,
                                               params.iouThreshold,
                                               params.minBoxSize,
                                               params.spatialScale,
                                               (const float*) iinfo,
                                               this->anchorsDev,
                                               nvinfer1::DataType::kFLOAT,
                                               NCHW,
                                               scores,
                                               nvinfer1::DataType::kFLOAT,
                                               NCHW,
                                               deltas,
                                               nvinfer1::DataType::kFLOAT,
                                               NCHW,
                                               fmap,
                                               workspace,
                                               nvinfer1::DataType::kFLOAT,
                                               rois,
                                               nvinfer1::DataType::kFLOAT,
                                               NCHW,
                                               pfmap);
    assert(status == STATUS_SUCCESS);
    return 0;
}

size_t RPROIPlugin::getSerializationSize() const
{
    size_t paramSize = sizeof(RPROIParams);
    size_t intSize = sizeof(int) * 4;
    size_t ratiosSize = sizeof(float) * params.anchorsRatioCount;
    size_t scalesSize = sizeof(float) * params.anchorsScaleCount;
    return paramSize + intSize + ratiosSize + scalesSize;
}

void RPROIPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    *reinterpret_cast<RPROIParams*>(d) = params;
    d += sizeof(RPROIParams);
    *reinterpret_cast<int*>(d) = A;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = C;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = H;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = W;
    d += sizeof(int);
    d += copyFromHost(d, anchorsRatiosHost, params.anchorsRatioCount);
    d += copyFromHost(d, anchorsScalesHost, params.anchorsScaleCount);
    assert(d == a + getSerializationSize());
}

float* RPROIPlugin::copyToHost(const void* srcHostData, int count)
{
    float* dstHostPtr = nullptr;
    CHECK(cudaMallocHost(&dstHostPtr, count * sizeof(float)));
    CHECK(cudaMemcpy(dstHostPtr, srcHostData, count * sizeof(float), cudaMemcpyHostToHost));
    return dstHostPtr;
}

int RPROIPlugin::copyFromHost(char* dstHostBuffer, const void* source, int count) const
{
    cudaMemcpy(dstHostBuffer, source, count * sizeof(float), cudaMemcpyHostToHost);
    return count * sizeof(float);
}

void RPROIPlugin::configureWithFormat(const Dims* inputDims, int nbInputs,
                                      const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int)
{
    assert(type == DataType::kFLOAT && format == PluginFormat::kNCHW);

    A = params.anchorsRatioCount * params.anchorsScaleCount;
    C = inputDims[2].d[0];
    H = inputDims[2].d[1];
    W = inputDims[2].d[2];

    assert(nbInputs == 4);
    assert(inputDims[0].d[0] == (2 * A) && inputDims[1].d[0] == (4 * A));
    assert(inputDims[0].d[1] == inputDims[1].d[1]
           && inputDims[0].d[1] == inputDims[2].d[1]);
    assert(inputDims[0].d[2] == inputDims[1].d[2]
           && inputDims[0].d[2] == inputDims[2].d[2]);
    assert(nbOutputs == 2
           && outputDims[0].nbDims == 3   // rois
           && outputDims[1].nbDims == 4); // pooled feature map
    assert(outputDims[0].d[0] == 1
           && outputDims[0].d[1] == params.nmsMaxOut
           && outputDims[0].d[2] == 4);
    assert(outputDims[1].d[0] == params.nmsMaxOut
           && outputDims[1].d[1] == C
           && outputDims[1].d[2] == params.poolingH
           && outputDims[1].d[3] == params.poolingW);

    return;
}

bool RPROIPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* RPROIPlugin::getPluginType() const
{
    return RPROI_PLUGIN_NAME;
}

const char* RPROIPlugin::getPluginVersion() const
{
    return RPROI_PLUGIN_VERSION;
}

void RPROIPlugin::terminate() {}

void RPROIPlugin::destroy() { delete this; }

IPluginV2* RPROIPlugin::clone() const
{
    IPluginV2* plugin = new RPROIPlugin(params, anchorsRatiosHost, anchorsScalesHost);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

RPROIPluginCreator::RPROIPluginCreator()
{

    mPluginAttributes.emplace_back(PluginField("poolingH", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("poolingW", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("featureStride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("preNmsTop", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("nmsMaxOut", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchorsRatioCount", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchorsScaleCount", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("minBoxSize", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("spatialScale", nullptr, PluginFieldType::kFLOAT32, 1));

    //TODO Do we need to pass the size attribute here for float arrarys, we
    //dont have that information at this point.
    mPluginAttributes.emplace_back(PluginField("anchorsRatios", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchorsScales", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

RPROIPluginCreator::~RPROIPluginCreator()
{
    //Free allocated memory (if any) here
}

const char* RPROIPluginCreator::getPluginName() const
{
    return RPROI_PLUGIN_NAME;
}

const char* RPROIPluginCreator::getPluginVersion() const
{
    return RPROI_PLUGIN_VERSION;
}

const PluginFieldCollection* RPROIPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* RPROIPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "poolingH"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            params.poolingH = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "poolingW"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            params.poolingW = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "featureStride"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            params.featureStride = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "preNmsTop"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            params.preNmsTop = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "nmsMaxOut"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            params.nmsMaxOut = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "anchorsRatioCount"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            params.anchorsRatioCount = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "anchorsScaleCount"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            params.anchorsScaleCount = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "iouThreshold"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            params.iouThreshold = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        if (!strcmp(attrName, "minBoxSize"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            params.minBoxSize = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        if (!strcmp(attrName, "spatialScale"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            params.spatialScale = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
        if (!strcmp(attrName, "anchorsRatios"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            anchorsRatios.reserve(params.anchorsRatioCount);
            const float* ratios = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < params.anchorsRatioCount; ++j)
            {
                anchorsRatios.push_back(*ratios);
                ratios++;
            }
        }
        if (!strcmp(attrName, "anchorsScales"))
        {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            anchorsScales.reserve(params.anchorsScaleCount);
            const float* scales = static_cast<const float*>(fields[i].data);
            for (int j = 0; j < params.anchorsScaleCount; ++j)
            {
                anchorsScales.push_back(*scales);
                scales++;
            }
        }
    }

    //This object will be deleted when the network is destroyed, which will
    //call RPROIPlugin::terminate()
    return new RPROIPlugin(params, anchorsRatios.data(), anchorsScales.data());
}

IPluginV2* RPROIPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    //This object will be deleted when the network is destroyed, which will
    //call RPROIPlugin::terminate()
    return new RPROIPlugin(serialData, serialLength);
}
