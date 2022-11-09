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
    try
    {
        constexpr int32_t seqLenKvPadded = 128;
        int32_t const batchSize = inputDesc[0].dims.d[0];
        int32_t const seqLenQ = inputDesc[0].dims.d[1];
        int32_t const seqLenKV = inputDesc[1].dims.d[1];
        int32_t const headNum = inputDesc[0].dims.d[2];
        int32_t const sizePerHead = inputDesc[0].dims.d[3];

        // Check for seq len to support dynamic input shape
        if (sizePerHead <= 64)
        {
            if (seqLenQ % 64 != 0)
            {
                gLogError << "Not support q buffer sequence length not multiple of 64 when head size < 64 for plugin "
                          << PLUGIN_NAME << " (q = " << seqLenQ << ", headSize = " << sizePerHead << ")" << std::endl;
                return 1;
            }
        }
        else if (sizePerHead <= 128)
        {
            if (seqLenQ % 32 != 0)
            {
                gLogError << "Not support q buffer sequence length not multiple of 32 when head size < 128 for plugin "
                          << PLUGIN_NAME << " (q = " << seqLenQ << ", headSize = " << sizePerHead << ")" << std::endl;
                return 1;
            }
        }
        else
        {
            if (seqLenQ % 16 != 0)
            {
                gLogError << "Not support q buffer sequence length not multiple of 16 for plugin " << PLUGIN_NAME
                          << " (q = " << seqLenQ << ", headSize = " << sizePerHead << ")" << std::endl;
                return 1;
            }
        }

        if (batchSize != m_.mOptBatchSize || m_.mOptSeqLenQ != seqLenQ)
        {
            m_.mOptSeqLenQ = initializeSeqlens(batchSize, seqLenQ, mCuSeqLensQ.get(), stream);
        }
        if (batchSize != m_.mOptBatchSize || m_.mOptSeqLenKV != seqLenKV)
        {
            m_.mOptSeqLenKV = initializeSeqlens(batchSize, seqLenKV, mCuSeqLensKV.get(), stream);
        }

        run_fmhca_api((void*) inputs[0], (void*) inputs[1], mCuSeqLensQ.get(), mCuSeqLensKV.get(), (void*) outputs[0],
            mSM, mKernels, static_cast<size_t>(batchSize), static_cast<size_t>(headNum),
            static_cast<size_t>(sizePerHead), static_cast<size_t>(seqLenQ), static_cast<size_t>(seqLenKvPadded),
            stream);

        return 0;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

REGISTER_TENSORRT_PLUGIN(fmhcaPluginCreator);
