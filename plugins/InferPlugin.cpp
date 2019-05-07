#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "checkMacrosPlugin.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
using namespace nvinfer1;
using namespace nvinfer1::plugin;

#include "NvPluginFasterRCNN.h"
#include "NvPluginSSD.h"
#include "clipPlugin.h"
#include "gridAnchorPlugin.h"
#include "lReluPlugin.h"
#include "nmsPlugin.h"
#include "normalizePlugin.h"
#include "nvPluginsLegacy.h"
#include "priorBoxPlugin.h"
#include "regionPlugin.h"
#include "reorgPlugin.h"
#include "batchedNMSPlugin.h"
#include "flattenConcat.h"

using nvinfer1::plugin::RPROIParams;
using nvinfer1::plugin::RPROIParamsLegacy;

namespace nvinfer1
{

namespace plugin
{

extern ILogger* gLogger;

// Instances of this class are statically constructed in initializePlugin.
// This ensures that each plugin is only registered a single time, as further calls to
// initializePlugin will be no-ops.
template <typename CreatorType>
class InitializePlugin{
    public:
        InitializePlugin(void* logger, const char* libNamespace) : mCreator{new CreatorType{}} {
            mCreator->setPluginNamespace(libNamespace);
            bool status = getPluginRegistry()->registerCreator(*mCreator, libNamespace);
            if (logger)
            {
                nvinfer1::plugin::gLogger = static_cast<nvinfer1::ILogger*>(logger);
                if (!status)
                {
                    std::string errorMsg{"Could not register plugin creator:  " + std::string(mCreator->getPluginName())
                    + " in namespace: " + std::string{mCreator->getPluginNamespace()}};
                    nvinfer1::plugin::gLogger->log(ILogger::Severity::kERROR, errorMsg.c_str());
                }
                else
                {
                    std::string verboseMsg{"Plugin Creator registration succeeded - " + std::string{mCreator->getPluginName()}};
                    nvinfer1::plugin::gLogger->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
                }
            }
        }

        InitializePlugin(const InitializePlugin&) = delete;
        InitializePlugin(InitializePlugin&&) = delete;
    private:
        std::unique_ptr<CreatorType> mCreator;
};

template <typename CreatorType>
void initializePlugin(void* logger, const char* libNamespace) {
    static InitializePlugin<CreatorType> plugin{logger, libNamespace};
}
//Legacy Plugin APIs

INvPlugin* createFasterRCNNPlugin(int featureStride, int preNmsTop, int nmsMaxOut,
                                  float iouThreshold, float minBoxSize, float spatialScale,
                                  nvinfer1::DimsHW pooling, nvinfer1::Weights anchorRatios,
                                  nvinfer1::Weights anchorScales)
{
    API_CHECK_RETVAL(anchorRatios.count > 0 && anchorScales.count > 0, nullptr);
    API_CHECK_RETVAL(pooling.d[0] > 0 && pooling.d[1] > 0, nullptr);
    return new RPROIPluginLegacy(RPROIParamsLegacy{pooling.d[0], pooling.d[1], featureStride, preNmsTop, nmsMaxOut, static_cast<int>(anchorRatios.count), static_cast<int>(anchorScales.count), iouThreshold, minBoxSize, spatialScale}, const_cast<float*>((const float*) anchorRatios.values), const_cast<float*>((const float*) anchorScales.values));
}

INvPlugin* createFasterRCNNPlugin(const void* data, size_t length)
{
    API_CHECK_RETVAL(length != 0, nullptr);
    return new RPROIPluginLegacy(data, length);
}

INvPlugin* createSSDNormalizePlugin(const nvinfer1::Weights* weights, bool acrossSpatial, bool channelShared, float eps)
{
    API_CHECK_RETVAL(weights[0].count >= 1, nullptr);
    return new NormalizeLegacy(weights, 1, acrossSpatial, channelShared, eps);
}

INvPlugin* createSSDNormalizePlugin(const void* data, size_t length)
{
    API_CHECK_RETVAL(length != 0, nullptr);
    return new NormalizeLegacy(data, length);
}

INvPlugin* createSSDPermutePlugin(Quadruple permuteOrder)
{
    std::cout << "WARNING: The Permute Plugin is being deprecated and will be removed in the next TensorRT release. Please use the TensorRT Shuffle layer for Permute operation" << std::endl;
    return new PermuteLegacy(permuteOrder);
}

INvPlugin* createSSDPermutePlugin(const void* data, size_t length)
{
    std::cout << "WARNING: The Permute Plugin is being deprecated and will be removed in the next TensorRT release. Please use the TensorRT Shuffle layer for Permute operation" << std::endl;
    API_CHECK_RETVAL(length != 0, nullptr);
    return new PermuteLegacy(data, length);
}

INvPlugin* createSSDPriorBoxPlugin(PriorBoxParameters param)
{
    API_CHECK_RETVAL(param.numMinSize > 0 && param.minSize != nullptr, nullptr);
    return new PriorBoxLegacy(param);
}

INvPlugin* createSSDPriorBoxPlugin(const void* data, size_t length)
{
    API_CHECK_RETVAL(length != 0, nullptr);
    return new PriorBoxLegacy(data, length);
}

INvPlugin* createSSDAnchorGeneratorPlugin(GridAnchorParameters* param, int numLayers)
{
    API_CHECK_RETVAL(numLayers > 0, nullptr);
    API_CHECK_RETVAL(param != nullptr, nullptr);
    return new GridAnchorGeneratorLegacy(param, numLayers);
}

INvPlugin* createSSDAnchorGeneratorPlugin(const void* data, size_t length)
{
    API_CHECK_RETVAL(length != 0, nullptr);
    API_CHECK_RETVAL(data != nullptr, nullptr);
    return new GridAnchorGeneratorLegacy(data, length);
}

INvPlugin* createSSDDetectionOutputPlugin(DetectionOutputParameters param)
{
    return new DetectionOutputLegacy(param);
}

INvPlugin* createSSDDetectionOutputPlugin(const void* data, size_t length)
{
    API_CHECK_RETVAL(length != 0, nullptr);
    return new DetectionOutputLegacy(data, length);
}

INvPlugin* createConcatPlugin(int concatAxis, bool ignoreBatch)
{
    API_CHECK_RETVAL(concatAxis == 1 || concatAxis == 2 || concatAxis == 3, nullptr);
    return new ConcatLegacy(concatAxis, ignoreBatch);
}

INvPlugin* createConcatPlugin(const void* data, size_t length)
{
    API_CHECK_RETVAL(length != 0, nullptr);
    return new ConcatLegacy(data, length);
}

INvPlugin* createPReLUPlugin(float negSlope)
{
    return new PReLULegacy(negSlope);
}

INvPlugin* createPReLUPlugin(const void* data, size_t length)
{
    API_CHECK_RETVAL(length != 0, nullptr);
    return new PReLULegacy(data, length);
}

INvPlugin* createYOLOReorgPlugin(int stride)
{
    API_CHECK_RETVAL(stride >= 0, nullptr);
    return new ReorgLegacy(stride);
}

INvPlugin* createYOLOReorgPlugin(const void* data, size_t length)
{
    API_CHECK_RETVAL(length != 0, nullptr);
    return new ReorgLegacy(data, length);
}

INvPlugin* createYOLORegionPlugin(RegionParameters params)
{
    return new RegionLegacy(params);
}

INvPlugin* createYOLORegionPlugin(const void* data, size_t length)
{
    API_CHECK_RETVAL(length != 0, nullptr);
    return new RegionLegacy(data, length);
}

} // namespace plugin
} // namespace nvinfer1
//New Plugin APIs

extern "C" {
IPluginV2* createRPNROIPlugin(int featureStride, int preNmsTop, int nmsMaxOut,
                              float iouThreshold, float minBoxSize, float spatialScale,
                              nvinfer1::DimsHW pooling, nvinfer1::Weights anchorRatios,
                              nvinfer1::Weights anchorScales)
{
    API_CHECK_RETVAL(anchorRatios.count > 0 && anchorScales.count > 0, nullptr);
    API_CHECK_RETVAL(pooling.d[0] > 0 && pooling.d[1] > 0, nullptr);
    return new RPROIPlugin(RPROIParams{pooling.d[0], pooling.d[1], featureStride, preNmsTop, nmsMaxOut, static_cast<int>(anchorRatios.count), static_cast<int>(anchorScales.count), iouThreshold, minBoxSize, spatialScale}, const_cast<float*>((const float*) anchorRatios.values), const_cast<float*>((const float*) anchorScales.values));
}

IPluginV2* createNormalizePlugin(const nvinfer1::Weights* weights, bool acrossSpatial, bool channelShared, float eps)
{
    API_CHECK_RETVAL(weights[0].count >= 1, nullptr);
    return new Normalize(weights, 1, acrossSpatial, channelShared, eps);
}

IPluginV2* createPriorBoxPlugin(PriorBoxParameters param)
{
    API_CHECK_RETVAL(param.numMinSize > 0 && param.minSize != nullptr, nullptr);
    return new PriorBox(param);
}

IPluginV2* createAnchorGeneratorPlugin(GridAnchorParameters* param, int numLayers)
{
    API_CHECK_RETVAL(numLayers > 0, nullptr);
    API_CHECK_RETVAL(param != nullptr, nullptr);
    return new GridAnchorGenerator(param, numLayers);
}

IPluginV2* createNMSPlugin(DetectionOutputParameters param)
{
    return new DetectionOutput(param);
}

IPluginV2* createLReLUPlugin(float negSlope)
{
    return new PReLU(negSlope);
}

IPluginV2* createReorgPlugin(int stride)
{
    API_CHECK_RETVAL(stride >= 0, nullptr);
    return new Reorg(stride);
}

IPluginV2* createRegionPlugin(RegionParameters params)
{
    return new Region(params);
}

IPluginV2* createClipPlugin(const char* layerName, float clipMin, float clipMax)
{
    return new ClipPlugin(layerName, clipMin, clipMax);
}

IPluginV2* createBatchedNMSPlugin(NMSParameters params)
{
    return new BatchedNMSPlugin(params);
}

bool initLibNvInferPlugins(void* logger, const char* libNamespace)
{
    initializePlugin<nvinfer1::plugin::GridAnchorPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::NMSPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::ReorgPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::RegionPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::ClipPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::LReluPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::PriorBoxPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::NormalizePluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::RPROIPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::BatchedNMSPluginCreator>(logger, libNamespace);
    initializePlugin<nvinfer1::plugin::FlattenConcatPluginCreator>(logger, libNamespace);
    return true;
}
} // extern "C"
