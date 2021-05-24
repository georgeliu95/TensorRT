/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "efficientNMSPlugin.h"
#include "efficientNMSInference.h"

using namespace nvinfer1;
using nvinfer1::plugin::EfficientNMSPlugin;
using nvinfer1::plugin::EfficientNMSParameters;

EfficientNMSPlugin::EfficientNMSPlugin(EfficientNMSParameters param)
    : mParam(param)
{
}

EfficientNMSPlugin::EfficientNMSPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mParam = read<EfficientNMSParameters>(d);
    ASSERT(d == a + length);
}

const char* EfficientNMSPlugin::getPluginType() const noexcept
{
    return EFFICIENT_NMS_PLUGIN_NAME;
}

const char* EfficientNMSPlugin::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_PLUGIN_VERSION;
}

int EfficientNMSPlugin::getNbOutputs() const noexcept
{
    return 5;
}

int EfficientNMSPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void EfficientNMSPlugin::terminate() noexcept {}

size_t EfficientNMSPlugin::getSerializationSize() const noexcept
{
    return sizeof(EfficientNMSParameters);
}

void EfficientNMSPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mParam);
    ASSERT(d == a + getSerializationSize());
}

void EfficientNMSPlugin::destroy() noexcept
{
    delete this;
}

void EfficientNMSPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* EfficientNMSPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType EfficientNMSPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // num_detections, detection_classes and detection_indices use integer outputs
    if (index == 0 || index == 3 || index == 4)
    {
        return nvinfer1::DataType::kINT32;
    }
    // All others should use the same datatype as the input
    return inputTypes[0];
}

IPluginV2DynamicExt* EfficientNMSPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new EfficientNMSPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs EfficientNMSPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        // Accepts two or three inputs
        // If two inputs: [0] boxes, [1] scores
        // If three inputs: [0] boxes, [1] scores, [2] anchors
        ASSERT(nbInputs == 2 || nbInputs == 3);

        ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());

        // Shape of boxes input should be
        // Constant shape: [batch_size, num_boxes, num_classes, 4] or [batch_size, num_boxes, 4]
        //           shareLocation ==              0               or          1
        // or
        // Dynamic shape: some dimension values may be -1
        ASSERT(inputs[0].nbDims == 3 || inputs[0].nbDims == 4);

        // Shape of scores input should be
        // Constant shape: [batch_size, num_boxes, num_classes]
        // or
        // Dynamic shape: some dimension values may be -1
        ASSERT(inputs[1].nbDims == 3);

        if (nbInputs == 2)
        {
            mParam.boxDecoder = false;
        }
        if (nbInputs == 3)
        {
            // Shape of anchors input should be
            // Constant shape: [1, numAnchors, 4] or [batch_size, numAnchors, 4]
            // or
            // Dynamic shape: some dimension values may be -1
            ASSERT(inputs[2].nbDims == 3);
            mParam.boxDecoder = true;
            mParam.shareAnchors = (inputs[2].d[0]->isConstant() && inputs[2].d[0]->getConstantValue() == 1);
        }

        ASSERT(inputs[0].d[1]->isConstant() && inputs[0].d[2]->isConstant());
        ASSERT(inputs[1].d[1]->isConstant() && inputs[1].d[2]->isConstant());
        if (nbInputs == 3)
        {
            ASSERT(inputs[2].d[1]->isConstant() && inputs[2].d[2]->isConstant());
        }
        if (inputs[0].nbDims == 4)
        {
            ASSERT(inputs[0].d[3]->isConstant());
            mParam.shareLocation = (inputs[0].d[2]->getConstantValue() == 1);
            mParam.numBoxElements
                = exprBuilder
                      .operation(DimensionOperation::kPROD,
                          *exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[1], *inputs[0].d[2]),
                          *inputs[0].d[3])
                      ->getConstantValue();
        }
        else
        {
            mParam.shareLocation = true;
            mParam.numBoxElements = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[1], *inputs[0].d[2])
                                        ->getConstantValue();
        }

        mParam.numClasses = inputs[1].d[2]->getConstantValue();
        mParam.numScoreElements
            = exprBuilder.operation(DimensionOperation::kPROD, *inputs[1].d[1], *inputs[1].d[2])->getConstantValue();

        DimsExprs out_dim;
        // num_detections
        if (outputIndex == 0)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(1);
        }
        // detection_boxes
        else if (outputIndex == 1)
        {
            out_dim.nbDims = 3;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
            out_dim.d[2] = exprBuilder.constant(4);
        }
        // detection_scores
        else if (outputIndex == 2)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
        }
        // detection_classes
        else if (outputIndex == 3)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
        }
        // detection_indices
        else if (outputIndex == 4)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = exprBuilder.operation(
                DimensionOperation::kPROD, *inputs[0].d[0], *exprBuilder.constant(mParam.numOutputBoxes));
            out_dim.d[1] = exprBuilder.constant(3);
        }

        return out_dim;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool EfficientNMSPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    ASSERT(nbInputs == 2 || nbInputs == 3);
    ASSERT(nbOutputs == 5);
    if (nbInputs == 2)
    {
        ASSERT(0 <= pos && pos < 7);
        mParam.boxDecoder = false;
    }
    if (nbInputs == 3)
    {
        ASSERT(0 <= pos && pos < 8);
        mParam.boxDecoder = true;
    }

    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    const int posOut = pos - nbInputs;

    // num_detections and detection_classes output: int
    if (posOut == 0 || posOut == 3 || posOut == 4)
    {
        return out[posOut].type == DataType::kINT32 && out[posOut].format == PluginFormat::kLINEAR;
    }

    // all other inputs/outputs: fp32 or fp16
    const bool consistentFloatPrecision = inOut[0].type == inOut[pos].type;
    return (inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kFLOAT)
        && inOut[pos].format == PluginFormat::kLINEAR && consistentFloatPrecision;
}

void EfficientNMSPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    try
    {
        ASSERT(nbInputs == 2 || nbInputs == 3);
        ASSERT(nbOutputs == 5);

        // Shape of boxes input should be
        // Constant shape: [batch_size, num_boxes, num_classes, 4] or [batch_size, num_boxes, 1, 4]
        //           shareLocation ==              0               or          1
        const int numLocClasses = mParam.shareLocation ? 1 : mParam.numClasses;
        ASSERT(in[0].desc.dims.nbDims == 3 || in[0].desc.dims.nbDims == 4);
        if (in[0].desc.dims.nbDims == 3)
        {
            ASSERT(in[0].desc.dims.d[2] == 4);
            mParam.shareLocation = true;
        }
        else
        {
            ASSERT(in[0].desc.dims.d[2] == numLocClasses);
            ASSERT(in[0].desc.dims.d[3] == 4);
            mParam.shareLocation = (in[0].desc.dims.d[2] == 1);
        }

        // Shape of scores input should be
        // Constant shape: [batch_size, num_boxes, num_classes] or [batch_size, num_boxes, num_classes, 1]
        ASSERT(in[1].desc.dims.nbDims == 3 || (in[1].desc.dims.nbDims == 4 && in[1].desc.dims.d[3] == 1));

        if (nbInputs == 2)
        {
            mParam.boxDecoder = false;
        }
        if (nbInputs == 3)
        {
            // Shape of anchors input should be
            // Constant shape: [1, numAnchors, 4] or [batch_size, numAnchors, 4]
            ASSERT(in[2].desc.dims.nbDims == 3);
            mParam.boxDecoder = true;
        }

        mParam.numBoxElements = in[0].desc.dims.d[1] * in[0].desc.dims.d[2] * in[0].desc.dims.d[3];
        mParam.numAnchors = in[0].desc.dims.d[1];
        mParam.numScoreElements = in[1].desc.dims.d[1] * in[1].desc.dims.d[2];
        mParam.numClasses = in[1].desc.dims.d[2];
        mParam.shareAnchors = (mParam.boxDecoder && in[2].desc.dims.d[0] == 1);

        mParam.datatype = in[0].desc.type;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

size_t EfficientNMSPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    EfficientNMSParameters p = mParam;
    p.batchSize = inputs[0].dims.d[0];
    return EfficientNMSWorkspaceSize(p);
}

int EfficientNMSPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        const void* const boxesInput = inputs[0];
        const void* const scoresInput = inputs[1];
        const void* const anchorsInput = mParam.boxDecoder ? inputs[2] : nullptr;

        void* numDetectionsOutput = outputs[0];
        void* nmsBoxesOutput = outputs[1];
        void* nmsScoresOutput = outputs[2];
        void* nmsClassesOutput = outputs[3];
        void* nmsIndicesOutput = outputs[4];

        mParam.batchSize = inputDesc[0].dims.d[0];

        pluginStatus_t status
            = EfficientNMSInference(mParam, boxesInput, scoresInput, anchorsInput, numDetectionsOutput, nmsBoxesOutput,
                nmsScoresOutput, nmsClassesOutput, nmsIndicesOutput, workspace, stream);
        return status;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}
