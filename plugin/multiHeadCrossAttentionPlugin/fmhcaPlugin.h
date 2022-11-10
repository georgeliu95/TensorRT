#pragma once
#ifndef _FMHCA_PLUGIN_
#define _FMHCA_PLUGIN_

#include "common/bertCommon.h"
#include "fmha_cross_attention/include/commonDatatype.h"
#include "fmha_cross_attention/include/fmha_cross_attention.h"

#include <NvInfer.h>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace
{
static const char* PLUGIN_NAME{"fMHCA"};
static const char* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{
namespace plugin
{

class fmhcaPlugin : public IPluginV2DynamicExt
{
private:
    const std::string mLayerName;
    std::string mNamespace;

    // scalar need copy
    struct
    {
        int32_t mOptBatchSize{};
        int32_t mOptSeqLenQ{};
        int32_t mOptSeqLenKV{};
        int32_t mMaxBatchSize{};
        DataType mDataType{DataType::kFLOAT};
    } m_;

public:
    fmhcaPlugin(const std::string& name)
        : mLayerName(name)
    {
        init();
    }

    fmhcaPlugin(const std::string& name, const void* data, size_t length)
        : mLayerName(name)
    {
        memcpy(&m_, data, sizeof(m_));
    }

    fmhcaPlugin() = delete;
    ~fmhcaPlugin() = default;

    void init(bool loadCubins = false)
    {
        try
        {
            mSM = bert::getSMVersion();

            // initialize seqlens buffer
            allocateSeqlens(m_.mMaxBatchSize);
            m_.mOptSeqLenQ = initializeSeqlens(m_.mOptBatchSize, m_.mOptSeqLenQ, mCuSeqLensQ.get());
            m_.mOptSeqLenKV = initializeSeqlens(m_.mOptBatchSize, m_.mOptSeqLenKV, mCuSeqLensKV.get());

            if (loadCubins)
            {
                createMHARunner();
            }
        }
        catch (const std::exception& e)
        {
            caughtError(e);
        }
    }

    void createMHARunner()
    {
        switch (m_.mDataType)
        {
        case DataType::kFLOAT: mKernels = getFMHCACubinKernels(plugin::DATA_TYPE_FP32, mSM); break;
        case DataType::kHALF: mKernels = getFMHCACubinKernels(plugin::DATA_TYPE_FP16, mSM); break;
        default: break;
        }
    }

    size_t getSerializationSize() const noexcept override
    {
        return sizeof(m_);
    }

    void serialize(void* buffer) const noexcept override
    {
        memcpy(buffer, &m_, sizeof(m_));
    }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        try
        {
            std::vector<char> buff;
            buff.resize(getSerializationSize());
            serialize(buff.data());

            auto p = new fmhcaPlugin(mLayerName, buff.data(), buff.size());
            p->mCuSeqLensQ = mCuSeqLensQ;
            p->mCuSeqLensKV = mCuSeqLensKV;
            p->setPluginNamespace(mNamespace.c_str());
            p->init(true);
            return p;
        }
        catch (const std::exception& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    // input0 q_packed in [b, s_q, h, d]
    // input1 kv_packed in [b, s_kv, h, 2, d]
    // output 0 in [b, s_q, h, d]
    DimsExprs getOutputDimensions(
        int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        DimsExprs out;
        out.nbDims = 4;
        out.d[0] = inputs[0].d[0];
        out.d[1] = inputs[0].d[1];
        out.d[2] = inputs[0].d[2];
        out.d[3] = inputs[0].d[3];
        return out;
    }

    bool supportsFormatCombination(
        int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        bool res = false;
        try
        {
            // load kernel and check if we have any implementations.
            auto hasImplement = [this](DataType dt)
            {
                switch (dt)
                {
                case DataType::kFLOAT: return getFMHCACubinKernels(plugin::DATA_TYPE_FP32, mSM)->isValid(/*dummy seq*/128);
                case DataType::kHALF: return getFMHCACubinKernels(plugin::DATA_TYPE_FP16, mSM)->isValid(/*dummy seq*/128);
                default: break;
                }
                return false;
            };

            if (inOut[pos].format != TensorFormat::kLINEAR)
            {
                return false;
            }

            switch (pos)
            {
            case 0: res = hasImplement(inOut[pos].type) && inOut[pos].dims.nbDims == 4; break;
            case 1: res = inOut[pos].type == inOut[0].type && inOut[pos].dims.nbDims == 5; break;
            case 2: res = inOut[pos].type == inOut[0].type && inOut[pos].dims.nbDims == 4; break;
            default: // should NOT be here
                break;
            }
        }
        catch (const std::exception& e)
        {
            caughtError(e);
        }
        return res;
    }

    void allocateSeqlens(int32_t maxBatchSize)
    {
        // allocate seqlens buffer
        auto allocBuffer = [this, &maxBatchSize](bert::cuda_shared_ptr<void>& dptr) {
            if (!dptr && maxBatchSize)
            {
                void* cudaMem{nullptr};
                PLUGIN_CHECK(cudaMalloc(&cudaMem, sizeof(int32_t) * (maxBatchSize + 1)));
                bert::make_cuda_shared(dptr, cudaMem);
            }
        };
        allocBuffer(mCuSeqLensQ);
        allocBuffer(mCuSeqLensKV);
        m_.mMaxBatchSize = maxBatchSize;
    }

    int32_t initializeSeqlens(int32_t b, int32_t s, void* cuSeqlensDev, cudaStream_t stream = 0)
    {
        if (!b || !s)
        {
            return s;
        }

        std::vector<int32_t> cuSeqlens(b + 1, 0);
        // Compute the prefix sum of the sequence lenghts.
        for (int32_t it = 0; it < b; it++)
        {
            cuSeqlens[it + 1] = cuSeqlens[it] + s;
        }

        PLUGIN_CUASSERT(cudaMemcpyAsync(
            cuSeqlensDev, cuSeqlens.data(), sizeof(int32_t) * cuSeqlens.size(), cudaMemcpyHostToDevice, stream));
        m_.mOptBatchSize = b;
        return s;
    }

    DataType getOutputDataType(
        int32_t outputIndex, const DataType* inputTypes, int32_t nbInputs) const noexcept override
    {
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs, const DynamicPluginTensorDesc* out,
        int32_t nbOutputs) noexcept override
    {
        try
        {
            int32_t const batchSize = in[0].max.d[0];
            int32_t const seqLenQ = in[0].max.d[1];
            int32_t const seqLenKV = in[1].max.d[1];

            allocateSeqlens(batchSize);
            if (batchSize != m_.mOptBatchSize || m_.mOptSeqLenQ != seqLenQ || m_.mOptSeqLenKV != seqLenKV)
            {
                m_.mOptSeqLenQ = initializeSeqlens(batchSize, seqLenQ, mCuSeqLensQ.get());
                m_.mOptSeqLenKV = initializeSeqlens(batchSize, seqLenKV, mCuSeqLensKV.get());
            }

            m_.mDataType = in[0].desc.type;
            createMHARunner();
        }
        catch (const std::exception& e)
        {
            caughtError(e);
        }
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,
        int32_t nbOutputs) const noexcept override
    {
        return 0;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        mNamespace = szNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }
    const char* getPluginType() const noexcept override
    {
        return PLUGIN_NAME;
    }
    const char* getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }
    int32_t initialize() noexcept override
    {
        return 0;
    }
    void terminate() noexcept override
    {
        return;
    }

    void destroy() noexcept override
    {
        delete this;
    }

    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

private:
    int32_t mSM{};
    bert::cuda_shared_ptr<void> mCuSeqLensQ;
    bert::cuda_shared_ptr<void> mCuSeqLensKV;
    FusedMultiHeadCrossAttentionKernel const* mKernels{};
}; // class fmhcaPlugin

class fmhcaPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;

public:
    fmhcaPluginCreator()
    {
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~fmhcaPluginCreator() {}

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        try
        {
            return new fmhcaPlugin(name);
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        try
        {
            auto p = new fmhcaPlugin(name, serialData, serialLength);
            p->init(true);
            return p;
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        mNamespace = szNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    const char* getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &mFC;
    }

}; // class fmhcaPluginCreator

} // namespace plugin
} // namespace nvinfer1

#endif
