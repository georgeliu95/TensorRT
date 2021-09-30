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
#ifndef TRT_INSTANCE_NORMALIZATION_3D_PLUGIN_H
#define TRT_INSTANCE_NORMALIZATION_3D_PLUGIN_H

#include "instance_norm_fwd.h"
#include "plugin.h"
#include "serialize.hpp"
#include <cuda_fp16.h>
#include <cudnn.h>
#include <iostream>
#include <string>
#include <vector>

typedef unsigned short half_type;

namespace nvinfer1
{
namespace plugin
{
using namespace instance_norm_impl;
class InstanceNormalization3DPlugin final : public nvinfer1::IPluginV2DynamicExt
{

public:
    InstanceNormalization3DPlugin(float epsilon, nvinfer1::Weights const& scale, nvinfer1::Weights const& bias,
        int32_t relu = 0, float alpha = 0.f);
    InstanceNormalization3DPlugin(float epsilon, const std::vector<float>& scale, const std::vector<float>& bias,
        int32_t relu = 0, float alpha = 0.f);
    InstanceNormalization3DPlugin(void const* serialData, size_t serialLength);

    InstanceNormalization3DPlugin() = delete;

    ~InstanceNormalization3DPlugin() override;

    int32_t getNbOutputs() const noexcept override;

    // DynamicExt plugins returns DimsExprs class instead of Dims
    using nvinfer1::IPluginV2::getOutputDimensions;
    DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;

    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    using nvinfer1::IPluginV2::getWorkspaceSize;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs,
        const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;

    using nvinfer1::IPluginV2::enqueue;
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    // DynamicExt plugin supportsFormat update.
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const
        noexcept override;

    using nvinfer1::IPluginV2Ext::configurePlugin;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;

private:
    float _epsilon;
    float _alpha;
    int32_t _relu;
    int32_t _nchan;
    std::vector<float> _h_scale;
    std::vector<float> _h_bias;
    float* _d_scale;
    float* _d_bias;
    cudnnHandle_t _cudnn_handle;
    cudnnTensorDescriptor_t _x_desc, _y_desc, _b_desc;
    const char* mPluginNamespace;
    std::string mNamespace;
    bool initialized{false};

    // NDHWC implementation
    InstanceNormFwdParams _params;
    InstanceNormFwdContext _context;

    float _in_scale;
    float _out_scale;
};

class InstanceNormalization3DPluginCreator : public BaseCreator
{
public:
    InstanceNormalization3DPluginCreator();

    ~InstanceNormalization3DPluginCreator() noexcept override = default;

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;

    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_INSTANCE_NORMALIZATION_3D_PLUGIN_H
