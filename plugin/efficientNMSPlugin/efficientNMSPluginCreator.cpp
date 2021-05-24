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

using namespace nvinfer1;
using nvinfer1::plugin::EfficientNMSPluginCreator;
using nvinfer1::plugin::EfficientNMSPlugin;
using nvinfer1::plugin::EfficientNMSParameters;

PluginFieldCollection EfficientNMSPluginCreator::mFC{};
std::vector<PluginField> EfficientNMSPluginCreator::mPluginAttributes;

EfficientNMSPluginCreator::EfficientNMSPluginCreator()
    : mParam{}
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_output_boxes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_selected_boxes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("background_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_sigmoid", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("box_coding", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EfficientNMSPluginCreator::getPluginName() const noexcept
{
    return EFFICIENT_NMS_PLUGIN_NAME;
}

const char* EfficientNMSPluginCreator::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_PLUGIN_VERSION;
}

const PluginFieldCollection* EfficientNMSPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* EfficientNMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "score_threshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.scoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.iouThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "max_output_boxes"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxes = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "max_selected_boxes"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numSelectedBoxes = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "background_class"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.backgroundClass = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_sigmoid"))
            {
                mParam.scoreSigmoid = *(static_cast<const bool*>(fields[i].data));
            }
            if (!strcmp(attrName, "box_coding"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.boxCoding = *(static_cast<const int*>(fields[i].data));
            }
        }

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

IPluginV2DynamicExt* EfficientNMSPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EfficientNMSPlugin::destroy()
        auto* plugin = new EfficientNMSPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
