/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "pyramidROIAlignTLTPlugin.h"
#include "plugin.h"
#include <cuda_runtime_api.h>

#include <fstream>

using namespace nvinfer1;
using namespace plugin;
using nvinfer1::plugin::PyramidROIAlignTLT;
using nvinfer1::plugin::PyramidROIAlignTLTPluginCreator;

namespace
{
const char* PYRAMIDROIALIGNTLT_PLUGIN_VERSION{"1"};
const char* PYRAMIDROIALIGNTLT_PLUGIN_NAME{"PyramidROIAlignTLT_TRT"};
} // namespace

PluginFieldCollection PyramidROIAlignTLTPluginCreator::mFC{};
std::vector<PluginField> PyramidROIAlignTLTPluginCreator::mPluginAttributes;

PyramidROIAlignTLTPluginCreator::PyramidROIAlignTLTPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("pooled_size", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* PyramidROIAlignTLTPluginCreator::getPluginName() const
{
    return PYRAMIDROIALIGNTLT_PLUGIN_NAME;
};

const char* PyramidROIAlignTLTPluginCreator::getPluginVersion() const
{
    return PYRAMIDROIALIGNTLT_PLUGIN_VERSION;
};

const PluginFieldCollection* PyramidROIAlignTLTPluginCreator::getFieldNames()
{
    return &mFC;
};

IPluginV2Ext* PyramidROIAlignTLTPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "pooled_size"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            mPooledSize = *(static_cast<const int*>(fields[i].data));
        }
    }
    return new PyramidROIAlignTLT(mPooledSize);
};

IPluginV2Ext* PyramidROIAlignTLTPluginCreator::deserializePlugin(const char* name, const void* data, size_t length)
{
    return new PyramidROIAlignTLT(data, length);
};

PyramidROIAlignTLT::PyramidROIAlignTLT(int pooled_size)
    : mPooledSize({pooled_size, pooled_size})
{

    assert(pooled_size > 0);
    // shape
    mInputHeight = TLTMaskRCNNConfig::IMAGE_SHAPE.d[1];
    mInputWidth = TLTMaskRCNNConfig::IMAGE_SHAPE.d[2];
    //Threshold to P3: Smaller -> P2
    mThresh = (224*224) / (4.0f);
};

int PyramidROIAlignTLT::getNbOutputs() const
{
    return 1;
};

int PyramidROIAlignTLT::initialize()
{
    return 0;
};

void PyramidROIAlignTLT::terminate(){

};

void PyramidROIAlignTLT::destroy()
{
    delete this;
};

size_t PyramidROIAlignTLT::getWorkspaceSize(int) const
{
    return 0;
}

bool PyramidROIAlignTLT::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
};

const char* PyramidROIAlignTLT::getPluginType() const
{
    return "PyramidROIAlignTLT_TRT";
};

const char* PyramidROIAlignTLT::getPluginVersion() const
{
    return "1";
};

IPluginV2Ext* PyramidROIAlignTLT::clone() const
{
    return new PyramidROIAlignTLT(*this);
};

void PyramidROIAlignTLT::setPluginNamespace(const char* libNamespace)
{
    mNameSpace = libNamespace;
};

const char* PyramidROIAlignTLT::getPluginNamespace() const
{
    return mNameSpace.c_str();
}

void PyramidROIAlignTLT::check_valid_inputs(const nvinfer1::Dims* inputs, int nbInputDims)
{
    // to be compatible with tensorflow node's input:
    // roi: [N, anchors, 4],
    // feature_map list(5 maps): p2, p3, p4, p5, p6
    assert(nbInputDims == 1 + mFeatureMapCount);

    nvinfer1::Dims rois = inputs[0];
    assert(rois.nbDims == 2);
    assert(rois.d[1] == 4);

    for (int i = 1; i < nbInputDims; ++i)
    {
        nvinfer1::Dims dims = inputs[i];

        // CHW with the same #C
        assert(dims.nbDims == 3 && dims.d[0] == inputs[1].d[0]);
    }
}

Dims PyramidROIAlignTLT::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{

    check_valid_inputs(inputs, nbInputDims);
    assert(index == 0);

    nvinfer1::Dims result;
    result.nbDims = 4;

    // mROICount
    result.d[0] = inputs[0].d[0];
    // mFeatureLength
    result.d[1] = inputs[1].d[0];
    // height
    result.d[2] = mPooledSize.y;
    // width
    result.d[3] = mPooledSize.x;

    return result;
};

int PyramidROIAlignTLT::enqueue(
    int batch_size, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    void* pooled = outputs[0];
    
    cudaError_t status = roiAlignTLT(stream, batch_size, mFeatureLength, mROICount, mThresh,

        mInputHeight, mInputWidth, inputs[0], &inputs[1], mFeatureSpatialSize,

        pooled, mPooledSize);

    assert(status == cudaSuccess);
    return 0;
};

size_t PyramidROIAlignTLT::getSerializationSize() const
{
    return sizeof(int) * 2 + sizeof(int) * 4 + sizeof(float) + sizeof(int) * 2 * mFeatureMapCount;
};

void PyramidROIAlignTLT::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mPooledSize.y);
    write(d, mPooledSize.x);
    write(d, mFeatureLength);
    write(d, mROICount);
    write(d, mInputHeight);
    write(d, mInputWidth);
    write(d, mThresh);
    for(int i = 0; i < mFeatureMapCount; i++)
    {
        write(d, mFeatureSpatialSize[i].y);
        write(d, mFeatureSpatialSize[i].x);
    }
    assert(d == a + getSerializationSize());
};

PyramidROIAlignTLT::PyramidROIAlignTLT(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mPooledSize = {read<int>(d), read<int>(d)};
    mFeatureLength = read<int>(d);
    mROICount = read<int>(d);
    mInputHeight = read<int>(d);
    mInputWidth = read<int>(d);
    mThresh = read<float>(d);
    for(int i = 0; i < mFeatureMapCount; i++)
    {
        mFeatureSpatialSize[i].y = read<int>(d);
        mFeatureSpatialSize[i].x = read<int>(d);
    }

    assert(d == a + length);
};

// Return the DataType of the plugin output at the requested index
DataType PyramidROIAlignTLT::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    // Only DataType::kFLOAT is acceptable by the plugin layer
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool PyramidROIAlignTLT::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool PyramidROIAlignTLT::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Configure the layer with input and output data types.
void PyramidROIAlignTLT::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    assert(supportsFormat(inputTypes[0], floatFormat));
    check_valid_inputs(inputDims, nbInputs);

    assert(nbOutputs == 1);
    assert(nbInputs == 1 + mFeatureMapCount);

    mROICount = inputDims[0].d[0];
    mFeatureLength = inputDims[1].d[0];

    for (size_t layer = 0; layer < mFeatureMapCount; ++layer)
    {
        mFeatureSpatialSize[layer] = {inputDims[layer + 1].d[1], inputDims[layer + 1].d[2]};
    }
}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void PyramidROIAlignTLT::attachToContext(
    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void PyramidROIAlignTLT::detachFromContext() {}
