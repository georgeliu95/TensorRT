#include "flattenConcat.h"
#include "NvInfer.h"
#include "checkMacrosPlugin.h"
#include "clip.h"
#include "plugin.h"

#include <algorithm>
#include <cstring>
#include <cudnn.h>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::FlattenConcat;
using nvinfer1::plugin::FlattenConcatPluginCreator;

static const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
static const char* FLATTENCONCAT_PLUGIN_NAME{"FlattenConcat_TRT"};

PluginFieldCollection FlattenConcatPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

FlattenConcat::FlattenConcat(int concatAxis, bool ignoreBatch)
    : mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
{
    ASSERT(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
}

FlattenConcat::FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis, const int* inputConcatAxis)
    : mIgnoreBatch(ignoreBatch)
    , mConcatAxisID(concatAxis)
    , mOutputConcatAxis(outputConcatAxis)
    , mNumInputs(numInputs)
{
    ASSERT(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    LOG_ERROR(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    for (int i = 0; i < mNumInputs; ++i)
    {
        mInputConcatAxis[i] = inputConcatAxis[i];
    }
}

FlattenConcat::FlattenConcat(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    mIgnoreBatch = read<bool>(d);
    mConcatAxisID = read<int>(d);
    ASSERT(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    mOutputConcatAxis = read<int>(d);
    mNumInputs = read<int>(d);
    LOG_ERROR(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
    LOG_ERROR(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(int)));

    std::for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs, [&](int& inp) { inp = read<int>(d); });

    mCHW = read<nvinfer1::DimsCHW>(d);

    std::for_each(mCopySize, mCopySize + mNumInputs, [&](size_t& inp) { inp = read<size_t>(d); });

    ASSERT(d == a + length);
}
FlattenConcat::~FlattenConcat()
{
    if (mInputConcatAxis)
    {
        LOG_ERROR(cudaFreeHost(mInputConcatAxis));
    }
    if (mCopySize)
    {
        LOG_ERROR(cudaFreeHost(mCopySize));
    }
}
int FlattenConcat::getNbOutputs() const { return 1; }

Dims FlattenConcat::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims >= 1);
    ASSERT(index == 0);
    
    mNumInputs = nbInputDims;
    LOG_ERROR(cudaMallocHost((void**) &mInputConcatAxis, nbInputDims * sizeof(int)));
    int outputConcatAxis = 0;

    for (int i = 0; i < nbInputDims; ++i)
    {
        int flattenInput = 0;
        ASSERT(inputs[i].nbDims == 3);
        if (mConcatAxisID != 1)
        {
            ASSERT(inputs[i].d[0] == inputs[0].d[0]);
        }
        if (mConcatAxisID != 2)
        {
            ASSERT(inputs[i].d[1] == inputs[0].d[1]);
        }
        if (mConcatAxisID != 3)
        {
            ASSERT(inputs[i].d[2] == inputs[0].d[2]);
        }
        flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
        outputConcatAxis += flattenInput;
    }

    return DimsCHW(mConcatAxisID == 1 ? outputConcatAxis : 1,
                   mConcatAxisID == 2 ? outputConcatAxis : 1,
                   mConcatAxisID == 3 ? outputConcatAxis : 1);
}

int FlattenConcat::initialize()
{
    LOG_ERROR(cublasCreate(&mCublas));
    return 0;
}

void FlattenConcat::terminate()
{
    LOG_ERROR(cublasDestroy(mCublas));
}

size_t FlattenConcat::getWorkspaceSize(int) const { return 0; }

int FlattenConcat::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int numConcats = 1;
    ASSERT(mConcatAxisID != 0);
    numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());
    cublasSetStream(mCublas, stream);

    if (!mIgnoreBatch)
    {
        numConcats *= batchSize;
    }

    auto* output = reinterpret_cast<float*>(outputs[0]);
    int offset = 0;
    for (int i = 0; i < mNumInputs; ++i)
    {
        const auto* input = reinterpret_cast<const float*>(inputs[i]);
        float* inputTemp;
        LOG_ERROR(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));

        LOG_ERROR(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize, cudaMemcpyDeviceToDevice, stream));

        for (int n = 0; n < numConcats; ++n)
        {
            LOG_ERROR(cublasScopy(mCublas, mInputConcatAxis[i],
                                  inputTemp + n * mInputConcatAxis[i], 1,
                                  output + (n * mOutputConcatAxis + offset), 1));
        }
        LOG_ERROR(cudaFree(inputTemp));
        offset += mInputConcatAxis[i];
    }

    return 0;
}

size_t FlattenConcat::getSerializationSize() const
{
    return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
}

void FlattenConcat::serialize(void* buffer) const
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
    for (int i = 0; i < mNumInputs; ++i)
    {
        write(d, mCopySize[i]);
    }
    ASSERT(d == a + getSerializationSize());
}

void FlattenConcat::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(nbOutputs == 1);
    mCHW = inputs[0];
    mNumInputs = nbInputs;
    ASSERT(inputs[0].nbDims == 3);
    
    for (int i = 0; i < nbInputs; ++i)
    {
        int flattenInput = 0;
        ASSERT(inputs[i].nbDims == 3);
        if (mConcatAxisID != 1)
        {
            ASSERT(inputs[i].d[0] == inputs[0].d[0]);
        }
        if (mConcatAxisID != 2)
        {
            ASSERT(inputs[i].d[1] == inputs[0].d[1]);
        }
        if (mConcatAxisID != 3)
        {
            ASSERT(inputs[i].d[2] == inputs[0].d[2]);
        }
        flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
        mInputConcatAxis[i] = flattenInput;
        mOutputConcatAxis += mInputConcatAxis[i];
    }
    LOG_ERROR(cudaMallocHost((void**) &mCopySize, nbInputs * sizeof(int)));
    for (int i = 0; i < nbInputs; ++i)
    {
        mCopySize[i] = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2] * sizeof(float);
    }
}

bool FlattenConcat::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}
const char* FlattenConcat::getPluginType() const { return "FlattenConcat_TRT"; }

const char* FlattenConcat::getPluginVersion() const { return "1"; }

void FlattenConcat::destroy() { delete this; }

IPluginV2* FlattenConcat::clone() const
{
    return new FlattenConcat(mConcatAxisID, mIgnoreBatch, mNumInputs, mOutputConcatAxis, mInputConcatAxis);
}

FlattenConcatPluginCreator::FlattenConcatPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FlattenConcatPluginCreator::getPluginName() const { return FLATTENCONCAT_PLUGIN_NAME; }

const char* FlattenConcatPluginCreator::getPluginVersion() const { return FLATTENCONCAT_PLUGIN_VERSION; }

const PluginFieldCollection* FlattenConcatPluginCreator::getFieldNames() { return &mFC; }

IPluginV2* FlattenConcatPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "axis"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mConcatAxisID = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "ignoreBatch"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            mIgnoreBatch = *(static_cast<const bool*>(fields[i].data));
        }
    }

    return new FlattenConcat(mConcatAxisID, mIgnoreBatch);
}

IPluginV2* FlattenConcatPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    //This object will be deleted when the network is destroyed, which will
    //call Concat::destroy()
    return new FlattenConcat(serialData, serialLength);
}
