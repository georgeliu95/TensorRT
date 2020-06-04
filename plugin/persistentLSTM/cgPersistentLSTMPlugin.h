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
#ifndef CG_PERSISTENT_LSTM_PLUGIN_H
#define CG_PERSISTENT_LSTM_PLUGIN_H

#ifdef __linux__
#if (defined(__x86_64__) || defined(__PPC__))

#include "NvInferPlugin.h"
#include "cgPersistentLSTM.h"
#include "cudaDriverWrapper.h"
#include "legacy_plugin.h"
#include <cassert>
#include <string>
#include <vector>

namespace nvinfer1
{
namespace plugin
{

class CgPersistentLSTMPlugin : public IPluginV2Ext
{
public:
    CgPersistentLSTMPlugin(CgPLSTMParameters params);

    CgPersistentLSTMPlugin(const void* data, size_t length);

    CgPersistentLSTMPlugin(int hiddenSize, int numLayers, int bidirectionFactor, int setInitialStates);

    ~CgPersistentLSTMPlugin() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    bool supportsFormat(DataType type, PluginFormat format) const override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    int initialize() override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    void terminate() override;

    IPluginV2Ext* clone() const override;

    void destroy() override;

    void setPluginNamespace(const char* libNamespace) override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mNamespace.c_str();
    }

    size_t getWorkspaceSize(int maxBatchSize) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    void detachFromContext() override;

private:
    std::string mNamespace;
    int maxBatchSize{0}, seqLength{0}, dataSize{0}, inputSize{0};
    nvinfer1::DataType dataType{nvinfer1::DataType::kHALF};
    CgPLSTMParameters param;

    CgPersistentLSTM* lstmRunner{nullptr};

    // cubin related
    AllocatedSizedRegion cubinOut;
    AllocatedSizedRegion loweredName;

    size_t sharedMemoryRequired{0};
    CUmodule module{};
    void _createCubin();
    CUDADriverWrapper wrap;
};

class CgPersistentLSTMPluginCreator : public IPluginCreator
{
public:
    CgPersistentLSTMPluginCreator();

    ~CgPersistentLSTMPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* libNamespace) override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    CgPLSTMParameters param;
};

} // namespace plugin
} // namespace nvinfer1

#endif // (defined(__x86_64__) || defined(__PPC__))
#endif //__linux__
#endif // CG_PERSISTENT_LSTM_PLUGIN_H
