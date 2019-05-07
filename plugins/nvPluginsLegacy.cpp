#include "nvPluginsLegacy.h"
#include "NvInfer.h"
#include "checkMacrosPlugin.h"
#include "plugin.h"
#include "rpnMacros.h"
#include "rpnlayer.h"
#include "ssd.h"
#include "yolo.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <vector>

using std::vector;
using nvinfer1::Dims;
using nvinfer1::Weights;
using nvinfer1::PluginType;
using nvinfer1::plugin::NormalizeLegacy;
using nvinfer1::plugin::PermuteLegacy;
using nvinfer1::plugin::PriorBoxLegacy;
using nvinfer1::plugin::DetectionOutputLegacy;
using nvinfer1::plugin::GridAnchorGeneratorLegacy;
using nvinfer1::plugin::ConcatLegacy;
using nvinfer1::plugin::RPROIPluginLegacy;
using nvinfer1::plugin::RPROIParamsLegacy;
using nvinfer1::plugin::PReLULegacy;
using nvinfer1::plugin::RegionLegacy;
using nvinfer1::plugin::RegionParameters;
using nvinfer1::plugin::ReorgLegacy;

// TRT-6104: Waive legacy plugins from coverage analysis.
// Reasons:
//     1. The samples and tests are using the new plugin API now.
//     2. The legacy plugins are going to be deprecated and removed in future TRT releases.
//     3. The legacy plugins and the new plugins only differ in API design. The underlying computation codes are the same.
//     4. This file results in 500 lines of uncovered codes and obscures other actual testing holes.
// GCOV_EXCL_START

RPROIPluginLegacy::RPROIPluginLegacy(RPROIParamsLegacy params, const float* anchorsRatios,
                                     const float* anchorsScales)
    : params(params)
{
    assert(params.anchorsRatioCount > 0 && params.anchorsScaleCount > 0);
    anchorsRatiosHost = copyToHost(anchorsRatios, params.anchorsRatioCount);
    anchorsScalesHost = copyToHost(anchorsScales, params.anchorsScaleCount);

    CHECK(cudaMalloc((void**) &anchorsDev, 4 * params.anchorsRatioCount * params.anchorsScaleCount * sizeof(float)));
    frcnnStatus_t status = generateAnchors(nullptr,
                                           params.anchorsRatioCount,
                                           anchorsRatiosHost,
                                           params.anchorsScaleCount,
                                           anchorsScalesHost,
                                           params.featureStride,
                                           anchorsDev);
    assert(status == STATUS_SUCCESS);
}

RPROIPluginLegacy::RPROIPluginLegacy(const void* data, size_t length)
    : anchorsDev(nullptr)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    params = *reinterpret_cast<const RPROIParamsLegacy*>(d);
    d += sizeof(RPROIParamsLegacy);
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
    frcnnStatus_t status = generateAnchors(nullptr,
                                           params.anchorsRatioCount,
                                           anchorsRatiosHost,
                                           params.anchorsScaleCount,
                                           anchorsScalesHost,
                                           params.featureStride,
                                           anchorsDev);
    assert(status == STATUS_SUCCESS);
}

RPROIPluginLegacy::~RPROIPluginLegacy()
{
    if (anchorsDev != nullptr)
        CHECK(cudaFree(anchorsDev));
    if (anchorsRatiosHost != nullptr)
        CHECK(cudaFreeHost(anchorsRatiosHost));
    if (anchorsScalesHost != nullptr)
        CHECK(cudaFreeHost(anchorsScalesHost));
}

void RPROIPluginLegacy::destroy()
{
    delete this;
}

void RPROIPluginLegacy::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int /*maxBatchSize*/)
{
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
}

int RPROIPluginLegacy::getNbOutputs() const
{
    return 2;
}

Dims RPROIPluginLegacy::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
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
    return DimsNCHW(params.nmsMaxOut, inputs[2].d[0], params.poolingH, params.poolingW);
}

size_t RPROIPluginLegacy::getWorkspaceSize(int maxBatchSize) const
{
    return RPROIInferenceFusedWorkspaceSize(maxBatchSize, A, H, W, params.nmsMaxOut);
}

int RPROIPluginLegacy::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
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

size_t RPROIPluginLegacy::getSerializationSize()
{
    size_t paramSize = sizeof(RPROIParamsLegacy);
    size_t intSize = sizeof(int) * 4;
    size_t ratiosSize = sizeof(float) * params.anchorsRatioCount;
    size_t scalesSize = sizeof(float) * params.anchorsScaleCount;
    return paramSize + intSize + ratiosSize + scalesSize;
}

void RPROIPluginLegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    *reinterpret_cast<RPROIParamsLegacy*>(d) = params;
    d += sizeof(RPROIParamsLegacy);
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

PluginType RPROIPluginLegacy::getPluginType() const
{
    return PluginType::kFASTERRCNN;
}

const char* RPROIPluginLegacy::getName() const
{
    return "FasterRCNN";
}

float* RPROIPluginLegacy::copyToHost(const void* srcHostData, int count)
{
    float* dstHostPtr = nullptr;
    CHECK(cudaMallocHost(&dstHostPtr, count * sizeof(float)));
    CHECK(cudaMemcpy(dstHostPtr, srcHostData, count * sizeof(float), cudaMemcpyHostToHost));
    return dstHostPtr;
}

int RPROIPluginLegacy::copyFromHost(char* dstHostBuffer, const void* source, int count)
{
    cudaMemcpy(dstHostBuffer, source, count * sizeof(float), cudaMemcpyHostToHost);
    return count * sizeof(float);
}

// NormalizeLegacy {{{
NormalizeLegacy::NormalizeLegacy(const Weights* weights, int nbWeights, bool acrossSpatial, bool channelShared,
                                 float eps)
    : acrossSpatial(acrossSpatial)
    , channelShared(channelShared)
    , eps(eps)
{
    assert(nbWeights == 1);
    assert(weights[0].count >= 1);
    mWeights = copyToDevice(weights[0].values, weights[0].count);
}

NormalizeLegacy::NormalizeLegacy(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    acrossSpatial = read<bool>(d);
    channelShared = read<bool>(d);
    eps = read<float>(d);

    int nbWeights = read<int>(d);
    mWeights = deserializeToDevice(d, nbWeights);
    assert(d == a + length);
}

NormalizeLegacy::~NormalizeLegacy()
{
    CUASSERT(cudaFree(const_cast<void*>(mWeights.values)));
}

void NormalizeLegacy::destroy()
{
    delete this;
}

int NormalizeLegacy::getNbOutputs() const
{
    return 1;
}

Dims NormalizeLegacy::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    assert(inputs[0].nbDims == 3);
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int NormalizeLegacy::initialize()
{
    CUBLASASSERT(cublasCreate(&mCublas));
    return 0;
}

void NormalizeLegacy::terminate()
{
    CUBLASASSERT(cublasDestroy(mCublas));
}

size_t NormalizeLegacy::getWorkspaceSize(int /*maxBatchSize*/) const
{
    return nvinfer1::plugin::normalizePluginWorkspaceSize(acrossSpatial, C, H, W);
}

int NormalizeLegacy::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    ssdStatus_t status = normalizeInference(stream, mCublas, acrossSpatial, channelShared,
                                            batchSize, C, H, W, eps,
                                            reinterpret_cast<const float*>(mWeights.values),
                                            inputData, outputData, workspace);
    assert(status == STATUS_SUCCESS);
    return 0;
}

size_t NormalizeLegacy::getSerializationSize()
{
    // C,H,W, acrossSpatial,channelShared, eps, mWeights.count,mWeights.values
    return sizeof(int) * 3 + sizeof(bool) * 2 + sizeof(float) + sizeof(int) + mWeights.count * sizeof(float);
}

void NormalizeLegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, acrossSpatial);
    write(d, channelShared);
    write(d, eps);
    write(d, (int) mWeights.count);
    serializeFromDevice(d, mWeights);

    assert(d == a + getSerializationSize());
}

void NormalizeLegacy::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int /*maxBatchSize*/)
{
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    if (channelShared)
    {
        assert(mWeights.count == 1);
    }
    else
    {
        assert(mWeights.count == C);
    }

    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    assert(inputDims[0].nbDims >= 1); // number of dimensions of the input tensor must be >=2
    assert(inputDims[0].d[0] == outputDims[0].d[0]
           && inputDims[0].d[1] == outputDims[0].d[1]
           && inputDims[0].d[2] == outputDims[0].d[2]);
}

Weights NormalizeLegacy::copyToDevice(const void* hostData, size_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

void NormalizeLegacy::serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
{
    CUASSERT(cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights NormalizeLegacy::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}

PluginType NormalizeLegacy::getPluginType() const { return PluginType::kNORMALIZE; }
const char* NormalizeLegacy::getName() const { return "Normalize"; }
// NormalizeLegacy }}}

// PermuteLegacy {{{
PermuteLegacy::PermuteLegacy(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    needPermute = read<bool>(d);
    permuteOrder = read<Quadruple>(d);
    oldSteps = read<Quadruple>(d);
    newSteps = read<Quadruple>(d);
    assert(d == a + length);
}

void PermuteLegacy::destroy()
{
    delete this;
}

int PermuteLegacy::getNbOutputs() const
{
    return 1;
}

Dims PermuteLegacy::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    assert(inputs[0].nbDims == 3);     // target 4D tensors for now
    assert(permuteOrder.data[0] == 0); // do not support permuting batch dimension for now
    for (int i = 0; i < 4; ++i)
    {
        int order = permuteOrder.data[i];
        assert(order < 4);
        if (i > 0)
        {
            assert(std::find(permuteOrder.data, permuteOrder.data + i, order) == permuteOrder.data + i && "There are duplicate orders");
        }
    }
    needPermute = false;
    for (int i = 0; i < 4; ++i)
    {
        if (permuteOrder.data[i] != i)
        {
            needPermute = true;
            break;
        }
    }
    if (needPermute)
    {
        return DimsCHW(inputs[0].d[permuteOrder.data[1] - 1],
                       inputs[0].d[permuteOrder.data[2] - 1],
                       inputs[0].d[permuteOrder.data[3] - 1]);
    }
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int PermuteLegacy::initialize()
{
    return 0;
}

void PermuteLegacy::terminate()
{
}

size_t PermuteLegacy::getWorkspaceSize(int /*maxBatchSize*/) const
{
    return 0;
}

int PermuteLegacy::enqueue(int batchSize, const void* const* inputs, void** outputs, void* /*workspace*/, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    const int count = newSteps.data[0] * batchSize;
    ssdStatus_t status = permuteInference(stream, needPermute, 4, count,
                                          &permuteOrder, &oldSteps, &newSteps,
                                          inputData, outputData);
    assert(status == STATUS_SUCCESS);
    return 0;
}

size_t PermuteLegacy::getSerializationSize()
{
    // needPermute, Quadruples(permuteOrder, oldSteps, newSteps)
    return sizeof(bool) + sizeof(Quadruple) * 3;
}

void PermuteLegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, needPermute);
    write(d, permuteOrder);
    write(d, oldSteps);
    write(d, newSteps);
    assert(d == a + getSerializationSize());
}

void PermuteLegacy::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int /*nbOutputs*/, int /*maxBatchSize*/)
{
    assert(nbInputs == 1);
    assert(inputDims[0].nbDims == 3); // target 4D tensors for now
    int C = inputDims[0].d[0], H = inputDims[0].d[1], W = inputDims[0].d[2];
    oldSteps = {{C * H * W, H * W, W, 1}};
    C = outputDims[0].d[0], H = outputDims[0].d[1], W = outputDims[0].d[2];
    newSteps = {{C * H * W, H * W, W, 1}};
}

PluginType PermuteLegacy::getPluginType() const { return PluginType::kPERMUTE; }
const char* PermuteLegacy::getName() const { return "Permute"; }
// PermuteLegacy }}}

// PriorBoxLegacy {{{
// TODO: this layer do not really use the data of the input tensors
// It only uses their dimensions
// So there's room for optimization by exploiting this removable dependency
PriorBoxLegacy::PriorBoxLegacy(PriorBoxParameters param)
    : param(param)
{
    assert(param.numMinSize > 0 && param.minSize != nullptr); // minSize is required!
    for (int i = 0; i < param.numMinSize; ++i)
    {
        assert(param.minSize[i] > 0 && "minSize must be positive");
    }
    minSize = copyToDevice(param.minSize, param.numMinSize);
    assert(param.numAspectRatios >= 0 && param.aspectRatios != nullptr);
    vector<float> tmpAR(1, 1);
    for (int i = 0; i < param.numAspectRatios; ++i)
    {
        float ar = param.aspectRatios[i];
        bool alreadyExist = false;
        for (unsigned j = 0; j < tmpAR.size(); ++j)
        {
            if (std::fabs(ar - tmpAR[j]) < 1e-6)
            {
                alreadyExist = true;
                break;
            }
        }
        if (!alreadyExist)
        {
            tmpAR.push_back(ar);
            if (param.flip)
            {
                tmpAR.push_back(1.0f / ar);
            }
        }
    }
    aspectRatios = copyToDevice(&tmpAR[0], tmpAR.size());
    numPriors = tmpAR.size() * param.numMinSize;
    if (param.numMaxSize > 0)
    {
        assert(param.numMinSize == param.numMaxSize && param.maxSize != nullptr);
        for (int i = 0; i < param.numMaxSize; ++i)
        {
            assert(param.maxSize[i] > param.minSize[i]
                   && "maxSize must be greater than minSize");
            numPriors++;
        }
        maxSize = copyToDevice(param.maxSize, param.numMaxSize);
    }
}

PriorBoxLegacy::PriorBoxLegacy(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    param = read<PriorBoxParameters>(d);
    numPriors = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    minSize = deserializeToDevice(d, param.numMinSize);
    if (param.numMaxSize > 0)
    {
        maxSize = deserializeToDevice(d, param.numMaxSize);
    }
    int numAspectRatios = read<int>(d);
    aspectRatios = deserializeToDevice(d, numAspectRatios);
    assert(d == a + length);
}

PriorBoxLegacy::~PriorBoxLegacy()
{
    CUASSERT(cudaFree(const_cast<void*>(minSize.values)));
    if (param.numMaxSize > 0)
    {
        CUASSERT(cudaFree(const_cast<void*>(maxSize.values)));
    }
    if (param.numAspectRatios > 0)
    {
        CUASSERT(cudaFree(const_cast<void*>(aspectRatios.values)));
    }
}

void PriorBoxLegacy::destroy()
{
    delete this;
}

int PriorBoxLegacy::getNbOutputs() const
{
    return 1;
}

Dims PriorBoxLegacy::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 2);
    assert(index == 0);
    // Particularity of the PriorBoxLegacy layer: no batchSize dimension needed
    H = inputs[0].d[1], W = inputs[0].d[2];
    // workaround for TRT
    return DimsCHW(2, H * W * numPriors * 4, 1);
}

int PriorBoxLegacy::initialize()
{
    return 0;
}

void PriorBoxLegacy::terminate()
{
}

size_t PriorBoxLegacy::getWorkspaceSize(int /*maxBatchSize*/) const
{
    return 0;
}

int PriorBoxLegacy::enqueue(int /*batchSize*/, const void* const* /*inputs*/, void** outputs, void* /*workspace*/, cudaStream_t stream)
{
    void* outputData = outputs[0];
    ssdStatus_t status = priorBoxInference(stream, param, H, W, numPriors,
                                           aspectRatios.count, minSize.values,
                                           maxSize.values, aspectRatios.values,
                                           outputData);
    assert(status == STATUS_SUCCESS);

    return 0;
}

size_t PriorBoxLegacy::getSerializationSize()
{
    // PriorBoxParameters, numPriors,H,W, minSize, maxSize, numAspectRatios, aspectRatios
    return sizeof(PriorBoxParameters) + sizeof(int) * 3 + sizeof(float) * (param.numMinSize + param.numMaxSize) + sizeof(int) + sizeof(float) * aspectRatios.count;
}

void PriorBoxLegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, param);
    write(d, numPriors);
    write(d, H);
    write(d, W);
    serializeFromDevice(d, minSize);
    if (param.numMaxSize > 0)
    {
        serializeFromDevice(d, maxSize);
    }
    write(d, (int) aspectRatios.count);
    serializeFromDevice(d, aspectRatios);
    assert(d == a + getSerializationSize());
}

void PriorBoxLegacy::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int /*maxBatchSize*/)
{
    assert(nbInputs == 2);
    assert(nbOutputs == 1);
    assert(inputDims[0].nbDims == 3);
    assert(inputDims[1].nbDims == 3);
    assert(outputDims[0].nbDims == 3);
    // H, W already set in getOutputDimensions
    assert(H == inputDims[0].d[1]);
    assert(W == inputDims[0].d[2]);
    // prepare for the inference function
    if (param.imgH == 0 || param.imgW == 0)
    {
        param.imgH = inputDims[1].d[1];
        param.imgW = inputDims[1].d[2];
    }
    if (param.stepH == 0 || param.stepW == 0)
    {
        param.stepH = static_cast<float>(param.imgH) / H;
        param.stepW = static_cast<float>(param.imgW) / W;
    }
    // unset unnecessary pointers
    param.minSize = nullptr;
    param.maxSize = nullptr;
    param.aspectRatios = nullptr;
}

Weights PriorBoxLegacy::copyToDevice(const void* hostData, size_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

void PriorBoxLegacy::serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
{
    cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights PriorBoxLegacy::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}

PluginType PriorBoxLegacy::getPluginType() const { return PluginType::kPRIORBOX; }
const char* PriorBoxLegacy::getName() const { return "PriorBox"; }

// PriorBoxLegacy }}}

GridAnchorGeneratorLegacy::GridAnchorGeneratorLegacy(GridAnchorParameters paramIn[], int
                                                                                         mNumLayers)
    : mNumLayers(mNumLayers)
{
    CUASSERT(cudaMallocHost((void**) &mNumPriors, mNumLayers * sizeof(int)));
    CUASSERT(cudaMallocHost((void**) &mDeviceWidths, mNumLayers * sizeof(Weights)));
    CUASSERT(cudaMallocHost((void**) &mDeviceHeights, mNumLayers * sizeof(Weights)));
    mParam.reserve(mNumLayers);
    for (int id = 0; id < mNumLayers; id++)
    {
        mParam[id] = paramIn[id];
        assert(mParam[id].numAspectRatios >= 0 && mParam[id].aspectRatios != nullptr);
        vector<float> tmpScales(mNumLayers + 1);

        for (int i = 0; i < mNumLayers; i++)
        {
            tmpScales[i] = (mParam[id].minSize
                            + (mParam[id].maxSize - mParam[id].minSize) * id / (mNumLayers - 1));
        }
        tmpScales.push_back(1.0f); // has 7 entries
        vector<float> scale0 = {0.1f, tmpScales[0], tmpScales[0]};

        vector<float> aspect_ratios;
        vector<float> scales;
        if (id == 0)
        {
            for (int i = 0; i < mParam[id].numAspectRatios; i++)
            {
                aspect_ratios.push_back(mParam[id].aspectRatios[i]);
                scales.push_back(scale0[i]);
            }
            mNumPriors[id] = mParam[id].numAspectRatios;
        }

        else
        {
            for (int i = 0; i < mParam[id].numAspectRatios; i++)
            {
                aspect_ratios.push_back(mParam[id].aspectRatios[i]);
            }
            aspect_ratios.push_back(1.0);

            //scales
            for (int i = 0; i < mParam[id].numAspectRatios; i++)
            {
                scales.push_back(tmpScales[id]);
            }
            auto scale_next = (id == mNumLayers - 1) ? 1.0 : (mParam[id].minSize + (mParam[id].maxSize - mParam[id].minSize) * (id + 1) / (mNumLayers - 1));
            scales.push_back(sqrt(tmpScales[id] * scale_next));

            mNumPriors[id] = mParam[id].numAspectRatios + 1;
        }

        vector<float> tmpWidths;
        vector<float> tmpHeights;
        for (int i = 0; i < mNumPriors[id]; i++)
        {
            float sqrt_AR = sqrt(aspect_ratios[i]);
            tmpWidths.push_back(scales[i] * sqrt_AR);
            tmpHeights.push_back(scales[i] / sqrt_AR);
        }

        mDeviceWidths[id] = copyToDevice(&tmpWidths[0], tmpWidths.size());
        mDeviceHeights[id] = copyToDevice(&tmpHeights[0], tmpHeights.size());
    }
}

GridAnchorGeneratorLegacy::GridAnchorGeneratorLegacy(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    mNumLayers = read<int>(d);
    CUASSERT(cudaMallocHost((void**) &mNumPriors, mNumLayers * sizeof(int)));
    CUASSERT(cudaMallocHost((void**) &mDeviceWidths, mNumLayers * sizeof(Weights)));
    CUASSERT(cudaMallocHost((void**) &mDeviceHeights, mNumLayers * sizeof(Weights)));
    mParam.reserve(mNumLayers);
    for (int id = 0; id < mNumLayers; id++)
    {
        mParam[id] = read<GridAnchorParameters>(d);
        mNumPriors[id] = read<int>(d);
        mDeviceWidths[id] = deserializeToDevice(d, mNumPriors[id]);
        mDeviceHeights[id] = deserializeToDevice(d, mNumPriors[id]);
    }
    assert(d == a + length);
}

GridAnchorGeneratorLegacy::~GridAnchorGeneratorLegacy()
{
    for (int id = 0; id < mNumLayers; id++)
    {
        CUERRORMSG(cudaFree(const_cast<void*>(mDeviceWidths[id].values)));
        CUERRORMSG(cudaFree(const_cast<void*>(mDeviceHeights[id].values)));
    }
    CUERRORMSG(cudaFreeHost(mNumPriors));
    CUERRORMSG(cudaFreeHost(mDeviceWidths));
    CUERRORMSG(cudaFreeHost(mDeviceHeights));
}

void GridAnchorGeneratorLegacy::destroy()
{
    delete this;
}

int GridAnchorGeneratorLegacy::getNbOutputs() const
{
    return mNumLayers;
}

Dims GridAnchorGeneratorLegacy::getOutputDimensions(int index, const Dims* /*inputs*/, int /*nbInputDims*/)
{
    // Particularity of the PriorBox layer: no batchSize dimension needed
    // 2 channels. First channel stores the mean of each prior coordinate.
    // Second channel stores the variance of each prior coordinate.
    // https://github.com/weiliu89/caffe/blob/ssd/src/caffe/layers/prior_box_layer.cpp
    return DimsCHW(2, mParam[index].H * mParam[index].W * mNumPriors[index] * 4, 1);
}

int GridAnchorGeneratorLegacy::initialize()
{
    return 0;
}

void GridAnchorGeneratorLegacy::terminate()
{
}

size_t GridAnchorGeneratorLegacy::getWorkspaceSize(int /*maxBatchSize*/) const
{
    return 0;
}

int GridAnchorGeneratorLegacy::enqueue(int /*batchSize*/, const void* const* /*inputs*/, void** outputs, void* /*workspace*/, cudaStream_t stream)
{
    for (int id = 0; id < mNumLayers; id++)
    {
        void* outputData = outputs[id];
        ssdStatus_t status = anchorGridInference(stream, mParam[id], mNumPriors[id],
                                                 mDeviceWidths[id].values, mDeviceHeights[id].values, outputData);
        assert(status == STATUS_SUCCESS);
    }

    return 0;
}

size_t GridAnchorGeneratorLegacy::getSerializationSize()
{
    // widths, heights, GridAnchorParameters, mNumPriors, mNumLayers
    int sum = 0;
    for (int i = 0; i < mNumLayers; i++)
    {
        sum += sizeof(float) * (mNumPriors[i] * 2);
    }
    return (sizeof(GridAnchorParameters) + sizeof(int)) * mNumLayers
        + sizeof(int) + sum;
}

void GridAnchorGeneratorLegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mNumLayers);
    for (int id = 0; id < mNumLayers; id++)
    {
        write(d, mParam[id]);
        write(d, mNumPriors[id]);
        serializeFromDevice(d, mDeviceWidths[id]);
        serializeFromDevice(d, mDeviceHeights[id]);
    }
    assert(d == a + getSerializationSize());
}

void GridAnchorGeneratorLegacy::configure(const Dims* /*inputDims*/, int /*nbInputs*/, const Dims* outputDims, int nbOutputs, int /*maxBatchSize*/)
{
    assert(nbOutputs == 6);
    assert(outputDims[0].nbDims == 3);
}

Weights GridAnchorGeneratorLegacy::copyToDevice(const void* hostData, size_t count)
{
    void* deviceData;
    CUASSERT(cudaMalloc(&deviceData, count * sizeof(float)));
    CUASSERT(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{DataType::kFLOAT, deviceData, int64_t(count)};
}

void GridAnchorGeneratorLegacy::serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
{
    cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights GridAnchorGeneratorLegacy::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}

PluginType GridAnchorGeneratorLegacy::getPluginType() const { return PluginType::kANCHORGENERATOR; }
const char* GridAnchorGeneratorLegacy::getName() const { return "GridAnchorGenerator"; }

// GridAnchorGeneratorLegacy }}}

// DetectOutput {{{
DetectionOutputLegacy::DetectionOutputLegacy(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    param = read<DetectionOutputParameters>(d);
    C1 = read<int>(d);
    C2 = read<int>(d);
    numPriors = read<int>(d);
    assert(d == a + length);
}

void DetectionOutputLegacy::destroy()
{
    delete this;
}

int DetectionOutputLegacy::getNbOutputs() const
{
    return 2;
}

int DetectionOutputLegacy::initialize()
{
    return 0;
}

void DetectionOutputLegacy::terminate()
{
}

Dims DetectionOutputLegacy::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 3);
    assert(index == 0 || index == 1);
    C1 = inputs[param.inputOrder[0]].d[0], C2 = inputs[param.inputOrder[1]].d[0];
    if (index == 0)
    {
        return DimsCHW(1, param.keepTopK, 7);
    }
#if 0 // FIXME: Why is this here?
    return DimsCHW(1, param.keepTopK, 1);
#else
    return DimsCHW(1, 1, 1);
#endif
}

size_t DetectionOutputLegacy::getWorkspaceSize(int maxBatchSize) const
{
    return detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, C1, C2, param.numClasses, numPriors, param.topK, DataType::kFLOAT, DataType::kFLOAT);
}

int DetectionOutputLegacy::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
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

size_t DetectionOutputLegacy::getSerializationSize()
{
    // DetectionOutputParameters, C1,C2,numPriors
    return sizeof(DetectionOutputParameters) + sizeof(int) * 3;
}

void DetectionOutputLegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, param);
    write(d, C1);
    write(d, C2);
    write(d, numPriors);
    assert(d == a + getSerializationSize());
}

void DetectionOutputLegacy::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int /*maxBatchSize*/)
{
    assert(nbInputs == 3);
    assert(nbOutputs == 2);
    assert(inputDims[0].nbDims == 3);
    assert(inputDims[1].nbDims == 3);
    assert(inputDims[2].nbDims == 3);
    assert(outputDims[0].nbDims == 3);
    assert(outputDims[1].nbDims == 3);
    // C1,C2 already set in getOutputDimensions
    assert(C1 == inputDims[param.inputOrder[0]].d[0]);
    assert(C2 == inputDims[param.inputOrder[1]].d[0]);
    numPriors = inputDims[param.inputOrder[2]].d[1] / 4;
    const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
    assert(numPriors * numLocClasses * 4 == inputDims[param.inputOrder[0]].d[0]);
    assert(numPriors * param.numClasses == inputDims[param.inputOrder[1]].d[0]);
}

PluginType DetectionOutputLegacy::getPluginType() const { return PluginType::kSSDDETECTIONOUTPUT; }
const char* DetectionOutputLegacy::getName() const { return "SSDDetectionOutput"; }

// DetectOutput }}}

// ConcatLegacy {{{
// TODO: write a kernel for this layer instead of using cuBLAS copy
ConcatLegacy::ConcatLegacy(int concatAxis, bool ignoreBatch)
    : mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
{
    // unable properly handle the case mConcatAxisID==0 for now
    // because the output dimension can not be set as (kN, C, H, W)
    assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
}

ConcatLegacy::ConcatLegacy(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    mIgnoreBatch = read<bool>(d);
    mConcatAxisID = read<int>(d);
    assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    mOutputConcatAxis = read<int>(d);
    mNumInputs = read<int>(d);
    CUASSERT(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    for (int i = 0; i < mNumInputs; ++i)
    {
        mInputConcatAxis[i] = read<int>(d);
    }
    mCHW = read<nvinfer1::DimsCHW>(d);
    assert(d == a + length);
}

ConcatLegacy::~ConcatLegacy()
{
    CUASSERT(cudaFreeHost(mInputConcatAxis));
}

void ConcatLegacy::destroy()
{
    delete this;
}

int ConcatLegacy::getNbOutputs() const
{
    return 1;
}

Dims ConcatLegacy::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims >= 1);
    assert(index == 0);
    mNumInputs = nbInputDims;
    CUASSERT(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    mOutputConcatAxis = 0;
#if 0
    //Use for debugging
    std::cout << " Concat nbInputs " << nbInputDims << "\n";
    std::cout << " Concat axis " << mConcatAxisID << "\n";
    for(int i=0; i<6; i++)
    {
        for(int j=0; j<3; j++)
            std::cout << " Concat InputDims[" << i << "]" << "d[" << j << "] is " <<
                inputs[i].d[j] << "\n";
    }
#endif
    for (int i = 0; i < nbInputDims; ++i)
    {
        assert(inputs[i].nbDims == 3);
        if (mConcatAxisID != 1)
        {
            assert(inputs[i].d[0] == inputs[0].d[0]);
        }
        if (mConcatAxisID != 2)
        {
            assert(inputs[i].d[1] == inputs[0].d[1]);
        }
        if (mConcatAxisID != 3)
        {
            assert(inputs[i].d[2] == inputs[0].d[2]);
        }
        mInputConcatAxis[i] = inputs[i].d[mConcatAxisID - 1];
        mOutputConcatAxis += mInputConcatAxis[i];
    }

    return DimsCHW(mConcatAxisID == 1 ? mOutputConcatAxis : inputs[0].d[0],
                   mConcatAxisID == 2 ? mOutputConcatAxis : inputs[0].d[1],
                   mConcatAxisID == 3 ? mOutputConcatAxis : inputs[0].d[2]);
}

int ConcatLegacy::initialize()
{
    CUBLASASSERT(cublasCreate(&mCublas));
    return 0;
}

void ConcatLegacy::terminate()
{
    CUBLASASSERT(cublasDestroy(mCublas));
}

size_t ConcatLegacy::getWorkspaceSize(int) const
{
    return 0;
}

int ConcatLegacy::enqueue(int batchSize, const void* const* inputs, void** outputs, void* /*workspace*/, cudaStream_t /*stream*/)
{
    int numConcats = 1;
    assert(mConcatAxisID != 0);
    for (int i = 0; i < mConcatAxisID - 1; ++i)
    {
        numConcats *= mCHW.d[i];
    }
    if (!mIgnoreBatch)
    {
        numConcats *= batchSize;
    }

    int concatSize = 1;
    for (int i = mConcatAxisID; i < 3; ++i)
    {
        concatSize *= mCHW.d[i];
    }

    auto* output = reinterpret_cast<float*>(outputs[0]);
    int offset = 0;
    for (int i = 0; i < mNumInputs; ++i)
    {
        const auto* input = reinterpret_cast<const float*>(inputs[i]);
        for (int n = 0; n < numConcats; ++n)
        {
            CUBLASASSERT(cublasScopy(mCublas, mInputConcatAxis[i] * concatSize,
                                 input + n * mInputConcatAxis[i] * concatSize, 1,
                                 output + (n * mOutputConcatAxis + offset) * concatSize, 1));
        }
        offset += mInputConcatAxis[i];
    }
    return 0;
}

size_t ConcatLegacy::getSerializationSize()
{
    // mIgnoreBatch, mConcatAxisID,mOutputConcatAxis,mNumInputs, mInputsConcatAxis[mNumInputs]
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims);
}

void ConcatLegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mIgnoreBatch);
    write(d, mConcatAxisID);
    write(d, mOutputConcatAxis);
    write(d, mNumInputs);
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mInputConcatAxis[i]);
    }
    write(d, mCHW);
    assert(d == a + getSerializationSize());
}

void ConcatLegacy::configure(const Dims* inputs, int /*nbInputs*/, const Dims* /*outputs*/, int nbOutputs, int)
{
    assert(nbOutputs == 1);
    mCHW = inputs[0];
}

PluginType ConcatLegacy::getPluginType() const { return PluginType::kCONCAT; }
const char* ConcatLegacy::getName() const { return "Concat"; }

// ConcatLegacy }}}

// LeakReLU {{{
PReLULegacy::PReLULegacy(float negSlope)
    : mNegSlope(negSlope)
    , mBatchDim(1)
{
}

PReLULegacy::PReLULegacy(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    mNegSlope = read<float>(d);
    mBatchDim = read<int>(d);
    assert(d == a + length);
}

int PReLULegacy::getNbOutputs() const { return 1; }

Dims PReLULegacy::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    return inputs[0];
}

int PReLULegacy::initialize() { return 0; }

void PReLULegacy::terminate() {}

size_t PReLULegacy::getWorkspaceSize(int /*maxBatchSize*/) const { return 0; }

int PReLULegacy::enqueue(int batchSize, const void* const* inputs, void** outputs, void* /*workspace*/, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    yoloStatus_t status = lReLUInference(stream, mBatchDim * batchSize, mNegSlope,
                                         inputData, outputData);
    assert(status == STATUS_SUCCESS);
    return 0;
}

size_t PReLULegacy::getSerializationSize()
{
    // mNegSlope, mBatchDim
    return sizeof(float) + sizeof(int);
}

void PReLULegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mNegSlope);
    write(d, mBatchDim);
    assert(d == a + getSerializationSize());
}

void PReLULegacy::configure(const Dims* inputDims, int nbInputs, const Dims* /*outputDims*/, int nbOutputs, int /*maxBatchSize*/)
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    mBatchDim = 1;
    for (int i = 0; i < inputDims[0].nbDims; ++i)
    {
        mBatchDim *= inputDims[0].d[i];
    }
}

PluginType PReLULegacy::getPluginType() const { return PluginType::kPRELU; }

const char* PReLULegacy::getName() const { return "PReLU"; }

void PReLULegacy::destroy() { delete this; }

// LeakReLU }}}

// ReorgLegacy {{{
ReorgLegacy::ReorgLegacy(int stride)
    : stride(stride)
{
}

ReorgLegacy::ReorgLegacy(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    stride = read<int>(d);
    assert(d == a + length);
}

int ReorgLegacy::getNbOutputs() const { return 1; }

Dims ReorgLegacy::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    return DimsCHW(inputs[0].d[0] * stride * stride, inputs[0].d[1] / stride, inputs[0].d[2] / stride);
}

int ReorgLegacy::initialize() { return 0; }

void ReorgLegacy::terminate() {}

size_t ReorgLegacy::getWorkspaceSize(int /*maxBatchSize*/) const { return 0; }

int ReorgLegacy::enqueue(int batchSize, const void* const* inputs, void** outputs, void* /*workspace*/, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    yoloStatus_t status = reorgInference(stream,
                                         batchSize, C, H, W, stride,
                                         inputData, outputData);
    assert(status == STATUS_SUCCESS);
    return 0;
}

size_t ReorgLegacy::getSerializationSize()
{
    // C, H, W, stride
    return sizeof(int) * 4;
}

void ReorgLegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, stride);
    assert(d == a + getSerializationSize());
}

void ReorgLegacy::configure(const Dims* inputDims, int nbInputs, const Dims* /*outputDims*/, int nbOutputs, int /*maxBatchSize*/)
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    assert(stride > 0);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    assert(H % stride == 0);
    assert(W % stride == 0);
}

PluginType ReorgLegacy::getPluginType() const { return PluginType::kYOLOREORG; }

const char* ReorgLegacy::getName() const { return "Reorg"; }

void ReorgLegacy::destroy() { delete this; }
// ReorgLegacy {{{

// RegionLegacy {{{

RegionLegacy::RegionLegacy(RegionParameters params)
    : num(params.num)
    , coords(params.coords)
    , classes(params.classes)
    , smTree(params.smTree)
{
}

RegionLegacy::RegionLegacy(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    num = read<int>(d);
    classes = read<int>(d);
    coords = read<int>(d);
    smTree = read<softmaxTree*>(d);
    assert(d == a + length);
}

int RegionLegacy::getNbOutputs() const { return 1; }

Dims RegionLegacy::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    return inputs[0];
}

int RegionLegacy::initialize() { return 0; }

void RegionLegacy::terminate() {}

size_t RegionLegacy::getWorkspaceSize(int /*maxBatchSize*/) const { return 0; }

int RegionLegacy::enqueue(int batchSize, const void* const* inputs, void** outputs, void* /*workspace*/, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    if (smTree != NULL)
    {
        hasSoftmaxTree = true;
    }
    else
    {
        hasSoftmaxTree = false;
    }
    yoloStatus_t status = regionInference(stream,
                                          batchSize, C, H, W,
                                          num, coords, classes,
                                          hasSoftmaxTree, smTree,
                                          inputData, outputData);
    assert(status == STATUS_SUCCESS);
    return 0;
}

size_t RegionLegacy::getSerializationSize()
{
    // C, H, W, num, classes, coords, *softmaxTree
    return sizeof(int) * 6 + sizeof(softmaxTree*);
}

void RegionLegacy::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, num);
    write(d, classes);
    write(d, coords);
    write(d, smTree);
    assert(d == a + getSerializationSize());
}

void RegionLegacy::configure(const Dims* inputDims, int nbInputs, const Dims* /*outputDims*/, int nbOutputs, int /*maxBatchSize*/)
{
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    assert(C == num * (coords + 1 + classes));
}

PluginType RegionLegacy::getPluginType() const { return PluginType::kYOLOREGION; }

const char* RegionLegacy::getName() const { return "Region"; }

void RegionLegacy::destroy() { delete this; }
// RegionLegacy }}}

// GCOV_EXCL_STOP
