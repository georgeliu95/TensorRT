#include "fmhcaPlugin.h"
#include "fmhca.h"

/*
   input[ 0]:  [float16],  [b, s_q,  h, d] // Q
   input[ 1]:  [float16],  [b, s_kv, h, 2, d] // K and V

   output[0]:  [float16],  (b, s_q, h, d)
   */

using namespace nvinfer1;
using namespace plugin;

PluginFieldCollection fmhcaPluginCreator::mFC{};
std::vector<PluginField> fmhcaPluginCreator::mPluginAttributes;

int32_t fmhcaPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int32_t result{-1};
    try
    {
        PLUGIN_ASSERT(mKernels);
        PLUGIN_ASSERT(mSM);
        PLUGIN_ASSERT(mCuSeqLensQ);
        PLUGIN_ASSERT(mCuSeqLensKV);

        constexpr int32_t seqLenKvPadded = 128;
        int32_t const batchSize = inputDesc[0].dims.d[0];
        int32_t const seqLenQ = inputDesc[0].dims.d[1];
        int32_t const seqLenKV = inputDesc[1].dims.d[1];
        int32_t const headNum = inputDesc[0].dims.d[2];
        int32_t const sizePerHead = inputDesc[0].dims.d[3];

        // Check for seq len to support dynamic input shape
        if (sizePerHead <= 64)
        {
            std::ostringstream oss;
            oss << "Not support q buffer sequence length not multiple of 64 when head size < 64 for plugin "
                << PLUGIN_NAME << " (q = " << seqLenQ << ", headSize = " << sizePerHead << ")" << std::endl;

            PLUGIN_VALIDATE(seqLenQ % 64 == 0, oss.str().c_str());
        }
        else if (sizePerHead <= 128)
        {
            std::ostringstream oss;
            oss << "Not support q buffer sequence length not multiple of 32 when head size < 128 for plugin "
                << PLUGIN_NAME << " (q = " << seqLenQ << ", headSize = " << sizePerHead << ")" << std::endl;

            PLUGIN_VALIDATE(seqLenQ % 32 == 0, oss.str().c_str());
        }
        else
        {
            std::ostringstream oss;
            oss << "Not support q buffer sequence length not multiple of 16 for plugin " << PLUGIN_NAME
                << " (q = " << seqLenQ << ", headSize = " << sizePerHead << ")" << std::endl;

            PLUGIN_VALIDATE(seqLenQ % 16 == 0, oss.str().c_str());
        }

        if (batchSize != m_.mOptBatchSize || m_.mOptSeqLenQ != seqLenQ || m_.mOptSeqLenKV != seqLenKV)
        {
            m_.mOptSeqLenQ = initializeSeqlens(batchSize, seqLenQ, mCuSeqLensQ.get(), stream);
            m_.mOptSeqLenKV = initializeSeqlens(batchSize, seqLenKV, mCuSeqLensKV.get(), stream);
        }

        result = run_fmhca_api((void*) inputs[0], (void*) inputs[1], mCuSeqLensQ.get(), mCuSeqLensKV.get(), (void*) outputs[0],
            mSM, mKernels, static_cast<size_t>(batchSize), static_cast<size_t>(headNum),
            static_cast<size_t>(sizePerHead), static_cast<size_t>(seqLenQ), static_cast<size_t>(seqLenKvPadded),
            stream);
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return result;
}

REGISTER_TENSORRT_PLUGIN(fmhcaPluginCreator);
