#ifdef __linux__
#ifdef __x86_64__

#include "cgPersistentLSTMPlugin.h"
#include "checkMacros.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::CgPersistentLSTMPlugin;
using nvinfer1::plugin::CgPersistentLSTMPluginCreator;

// For LSTM, the followings are needed:
// Input: x_data(NCHW, needs SNE transpose), sequenceLengths; weights(broadcast), bias (broadcast), init_h (batch x
// numLayers x hidden, needs transpose as well), init_c (batch x numLayers xhidden, needs transpose as well)
// Output: y, hidden size, cell state
// Workspace: tmp_i, tmp_h

PluginFieldCollection CgPersistentLSTMPluginCreator::mFC{};
std::vector<PluginField> CgPersistentLSTMPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(CgPersistentLSTMPluginCreator);

namespace
{
const char* CG_PERSISTENT_LSTM_PLUGIN_VERSION{"1"};
const char* CG_PERSISTENT_LSTM_PLUGIN_NAME{"CgPersistentLSTMPlugin_TRT"};
} // namespace

CgPersistentLSTMPlugin::CgPersistentLSTMPlugin(CgPLSTMParameters params)
    : param(params)
{
    lstmRunner = nullptr;
}

CgPersistentLSTMPlugin::CgPersistentLSTMPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char *>(data), *a = d;
    param = read<CgPLSTMParameters>(d);
    maxBatchSize = read<int>(d);
    seqLength = read<int>(d);
    dataSize = read<int>(d);
    inputSize = read<int>(d);
    dataType = read<nvinfer1::DataType>(d);
    sharedMemoryRequired = read<size_t>(d);
    auto cuSizeOut = read<size_t>(d);
    if (cuSizeOut > 0)
    {
        cubinOut.assignNewSpace(cuSizeOut);
        memcpy(cubinOut.data, d, cuSizeOut);
        d += cuSizeOut;
    }

    auto loweredNameSize = read<size_t>(d);
    if (loweredNameSize > 0)
    {
        loweredName.assignNewSpace(loweredNameSize);
        memcpy(loweredName.data, d, loweredNameSize);
        d += loweredNameSize;
    }

    assert(d == a + length);
    lstmRunner = nullptr;
}

// Default parameters
CgPersistentLSTMPlugin::CgPersistentLSTMPlugin(int hiddenSize, int numLayers, int bidirectionFactor, int setInitialStates)
{
    int device;
    CUASSERT(cudaGetDevice(&device));
    cudaDeviceProp deviceProp{};
    CUASSERT(cudaGetDeviceProperties(&deviceProp, device));

    param.setInitialStates = setInitialStates == 1;
    // default parameters
    param.gridSize = deviceProp.multiProcessorCount;

    param.hiddenSize = hiddenSize;
    param.numLayers = numLayers;

    param.FRAG_M = 32;
    param.FRAG_N = 8;
    param.FRAG_K = 16;

    param.isBi = bidirectionFactor == 2;
    // How many batch examples to do in a single loop iteraiton.
    param.innerStepSize = 8;

    if (deviceProp.multiProcessorCount == 40)
    {
        param.warpsPerBlockK = 1;
        param.warpsPerBlockM = 4;
        param.blockSplitKFactor = 2;

        param.unrollSplitK = 0;
        param.unrollGemmBatch = 0;

        param.gridSize = 64;
        param.blockSize = 128;
        param.rfSplitFactor = 6;

        if (param.isBi)
        {
            param.rfSplitFactor = 8;
            param.separatePath = true;
        }
    }

    else if (deviceProp.multiProcessorCount == 72)
    {
        if (param.isBi)
        {
            param.warpsPerBlockK = 2;
            param.warpsPerBlockM = 2;
            param.blockSplitKFactor = 2;

            param.unrollSplitK = 0;
            param.unrollGemmBatch = 0;

            param.gridSize = 128;
            param.blockSize = 128;
            param.rfSplitFactor = 12;
            param.separatePath = true;
        }
        else
        {
            param.warpsPerBlockK = 2;
            param.warpsPerBlockM = 4;
            param.blockSplitKFactor = 2;

            param.unrollSplitK = 1;
            param.unrollGemmBatch = 0;

            param.gridSize = 64;
            param.blockSize = 256;
            param.rfSplitFactor = 11;
            param.separatePath = true;
        }

    }

    else
    {
        param.warpsPerBlockK = 2;
        param.warpsPerBlockM = 4;
        param.blockSplitKFactor = 2;

        param.unrollSplitK = 1;
        param.unrollGemmBatch = 0;

        param.blockSize = 256;
        param.rfSplitFactor = 1;

        // more grid, less work per block (already doing two passes)
        if (param.isBi)
        {
            param.gridSize *= 2;
            param.blockSize = 128;
            param.warpsPerBlockK = 1;
            param.unrollSplitK = 0;
            param.blockSplitKFactor *= 2;
            param.rfSplitFactor = 1;
            param.separatePath = true;
        }

    }

    lstmRunner = nullptr;
}

// y, hidden state, cell state
int CgPersistentLSTMPlugin::getNbOutputs() const { return 3; }

Dims CgPersistentLSTMPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 6);
    assert(index >= 0 && index < this->getNbOutputs());
    assert((inputs[0].nbDims == 2) || (inputs[0].nbDims == 3));
    assert(inputs[1].nbDims == 1);
    // The dims of weights, I guess, does not really matter

    if (param.setInitialStates)
    {
        assert((inputs[4].nbDims == 2) || (inputs[4].nbDims == 3));
        assert((inputs[5].nbDims == 2) || (inputs[5].nbDims == 3));
    }

    seqLength = inputs[0].d[0];
    int bidirectionFactor = param.isBi ? 2 : 1;
    if (index == 0)
    {
        Dims dim0{};
        dim0.nbDims = inputs[0].nbDims;
        for (int i = 0; i < inputs[0].nbDims - 1; i++)
        {
            dim0.d[i] = inputs[0].d[i];
        }
        dim0.d[inputs[0].nbDims - 1] = bidirectionFactor * param.hiddenSize;
        return dim0;
    }

    Dims dim1{};

    if (param.setInitialStates)
    {
        dim1.nbDims = inputs[4].nbDims;
        for (int i = 0; i < inputs[4].nbDims; i++)
        {
            dim1.d[i] = inputs[4].d[i];
        }
    }
    else
    {
        dim1.nbDims = 3;
        dim1.d[0] = 1;
        dim1.d[1] = bidirectionFactor*param.numLayers;
        dim1.d[2] = param.hiddenSize;
    }

    return dim1;
}

// Currently only support half precision tensorcore
bool CgPersistentLSTMPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kHALF || (type == DataType::kINT32)) && (format == PluginFormat::kNCHW));
}

nvinfer1::DataType CgPersistentLSTMPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

// Input: x_data, rMat, wMat, init_h, init_c
// Output: y, hidden size, cell state
// Workspace: tmp_i, tmp_h
void CgPersistentLSTMPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
    if (param.setInitialStates)
    {
        assert(nbInputs == 6);
    }
    else
    {
        assert(nbInputs == 4);
    }

    assert(nbOutputs == 3);
    assert(std::none_of(outputIsBroadcast, outputIsBroadcast + nbOutputs, [](bool b) { return b; }));

    assert(inputTypes[0] == DataType::kHALF);
    assert(inputTypes[1] == DataType::kINT32);
    assert(inputTypes[2] == DataType::kHALF);
    assert(inputTypes[3] == DataType::kHALF);
    if (param.setInitialStates)
    {
        assert(inputTypes[4] == DataType::kHALF);
        assert(inputTypes[5] == DataType::kHALF);
    }

    // The dims of rMat and wMat, I guess, does not really matter
    assert((inputDims[0].nbDims == 2) || (inputDims[0].nbDims == 3));
    assert(inputDims[1].nbDims == 1);
    assert(outputDims[0].nbDims == inputDims[0].nbDims);

    if (param.setInitialStates)
    {
        assert((inputDims[4].nbDims == 2) || (inputDims[4].nbDims == 3));
        assert((inputDims[5].nbDims == 2) || (inputDims[5].nbDims == 3));
        assert(outputDims[1].nbDims == inputDims[4].nbDims);
        assert(outputDims[2].nbDims == inputDims[4].nbDims);
    }

    dataSize = inputTypes[0] == nvinfer1::DataType::kFLOAT ? 4 : 2; // only float and half precision
    dataType = inputTypes[0];

    seqLength = inputDims[0].d[0];
    this->maxBatchSize = maxBatchSize; // batchsize x z dimension (NZ), not for this testcase
    inputSize = inputDims[0].d[1];
    if (inputDims[0].nbDims == 3)
    {
        this->maxBatchSize *= inputDims[0].d[0];
        seqLength = inputDims[0].d[1];
        inputSize = inputDims[0].d[2];
    }
    _createCubin();
}

int CgPersistentLSTMPlugin::enqueue(
    int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    if (param.setInitialStates)
    {
        int bidirectionFactor = param.isBi ? 2 : 1;
        const void* x = inputs[0];
        const void* sequence = inputs[1];
        const void* rMat = inputs[2];
        const void* wMat = static_cast<const void*>(static_cast<const char*>(inputs[2])
            + bidirectionFactor * param.hiddenSize * param.hiddenSize * 4 * param.numLayers * dataSize);
        const void* bias = inputs[3];
        const void* hx = inputs[4];
        const void* cx = inputs[5];

        void* y = outputs[0];
        void* hy = outputs[1];
        void* cy = outputs[2];

        lstmRunner->execute(x, y, hx, cx, hy, cy, rMat, wMat, bias, sequence, batchSize, workspace, stream);
        return 0;
    }

    int bidirectionFactor = param.isBi ? 2 : 1;
    const void* x = inputs[0];
    const void* sequence = inputs[1];
    const void* rMat = inputs[2];
    const void* wMat = static_cast<const void*>(static_cast<const char*>(inputs[2])
        + bidirectionFactor * param.hiddenSize * param.hiddenSize * 4 * param.numLayers * dataSize);
    const void* bias = inputs[3];

    void* y = outputs[0];
    void* hy = outputs[1];
    void* cy = outputs[2];

    lstmRunner->execute(x, y, nullptr, nullptr, hy, cy, rMat, wMat, bias, sequence, batchSize, workspace, stream);
    return 0;
}

IPluginV2Ext* CgPersistentLSTMPlugin::clone() const
{
    auto* plugin = new CgPersistentLSTMPlugin(param);
    plugin->maxBatchSize = maxBatchSize;
    plugin->seqLength = seqLength;
    plugin->dataSize = dataSize;
    plugin->dataType = dataType;
    plugin->inputSize = inputSize;

    if (cubinOut.size > 0)
    {
        (plugin->cubinOut).assignNewSpace(cubinOut.size);
        memcpy((plugin->cubinOut).data, cubinOut.data, cubinOut.size);
    }

    if (loweredName.size > 0)
    {
        (plugin->loweredName).assignNewSpace(loweredName.size);
        memcpy((plugin->loweredName).data, loweredName.data, loweredName.size);
    }

    plugin->sharedMemoryRequired = sharedMemoryRequired;
    plugin->module = module;
    if (lstmRunner != nullptr)
    {
        plugin->initialize();
    }
    plugin->setPluginNamespace(mNamespace.c_str());

    return plugin;
}

// configurePlugin -> initialize -> getWorkspaceSize
size_t CgPersistentLSTMPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return nvinfer1::plugin::CgPersistentLSTM::computeGpuSize(maxBatchSize, seqLength, inputSize, dataSize, param);
}

bool CgPersistentLSTMPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool CgPersistentLSTMPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return (inputIndex == 2) || (inputIndex == 3);
}

size_t CgPersistentLSTMPlugin::getSerializationSize() const
{
    // CgPLSTMParameters, maxBatchSize, seqLength, dataSize, inputSize, dataType, cuSizeOut, sharedMemoryRequired,
    // cubinOut, loweredName
    return sizeof(CgPLSTMParameters) + sizeof(int) * 4 + sizeof(nvinfer1::DataType) + sizeof(size_t) * 3 + cubinOut.size
        + loweredName.size;
}

void CgPersistentLSTMPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, param);
    write(d, maxBatchSize);
    write(d, seqLength);
    write(d, dataSize);
    write(d, inputSize);
    write(d, dataType);
    write(d, sharedMemoryRequired);

    write(d, cubinOut.size);
    if (cubinOut.size > 0)
    {
        memcpy(d, cubinOut.data, cubinOut.size);
        d += cubinOut.size;
    }

    write(d, loweredName.size);
    if (loweredName.size > 0)
    {
        memcpy(d, loweredName.data, loweredName.size);
        d += loweredName.size;
    }
    assert(d == a + getSerializationSize());
}

const char* CgPersistentLSTMPlugin::getPluginType() const
{
    return CG_PERSISTENT_LSTM_PLUGIN_NAME;
}

const char* CgPersistentLSTMPlugin::getPluginVersion() const
{
    return CG_PERSISTENT_LSTM_PLUGIN_VERSION;
}

CgPersistentLSTMPluginCreator::CgPersistentLSTMPluginCreator()

{
    mPluginAttributes.emplace_back(PluginField("hiddenSize", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numLayers", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("bidirectionFactor", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("setInitialStates", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* CgPersistentLSTMPluginCreator::getPluginName() const { return CG_PERSISTENT_LSTM_PLUGIN_NAME; }

const char* CgPersistentLSTMPluginCreator::getPluginVersion() const { return CG_PERSISTENT_LSTM_PLUGIN_VERSION; }

const PluginFieldCollection* CgPersistentLSTMPluginCreator::getFieldNames() { return &mFC; }

IPluginV2* CgPersistentLSTMPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    int hiddenSize = 0;
    int numLayers = 0;
    int bidirectionFactor = 0;
    int setInitialStates=0;
    for (int i = 0; i < fc->nbFields; i++)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "hiddenSize"))
        {
            hiddenSize = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "numLayers"))
        {
            numLayers = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "bidirectionFactor"))
        {
            bidirectionFactor = *(static_cast<const int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "setInitialStates"))
        {
            setInitialStates = *(static_cast<const int*>(fields[i].data));
        }
    }

    return new CgPersistentLSTMPlugin(hiddenSize, numLayers, bidirectionFactor, setInitialStates);
}

IPluginV2* CgPersistentLSTMPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength)
{
    return new CgPersistentLSTMPlugin(serialData, serialLength);
}

#endif// __x86_64__
#endif //__linux__
