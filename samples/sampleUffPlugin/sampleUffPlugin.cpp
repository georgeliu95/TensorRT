/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cassert>
#include <chrono>
#include <cudnn.h>
#include <iostream>
#include <map>
#include <string.h>
#include <unordered_map>
#include <vector>

#include "NvUtils.h"
#include "argsParser.h"

using namespace nvuffparser;
using namespace nvinfer1;

#include "common.h"
#include "logger.h"

const std::string gSampleName = "TensorRT.sample_uff_plugin";
samplesCommon::Args gArgs;

constexpr size_t MAX_WORKSPACE = 1_GB;

inline int64_t volume(const Dims& d)
{
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}

inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
    case DataType::kINT32:
    // Fallthrough, same as kFLOAT
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}

static const int INPUT_H = 28;
static const int INPUT_W = 28;

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(locateFile(filename, gArgs.dataDirs), buffer, INPUT_H, INPUT_W);
}

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        gLogError << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

void* createMnistCudaBuffer(int64_t eltCount, DataType dtype, int num)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W */
    assert(eltCount == INPUT_H * INPUT_W);
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    std::vector<float> inputs(eltCount);

    /* read PGM file */
    uint8_t fileData[INPUT_H * INPUT_W];
    readPGMFile(std::to_string(num) + ".pgm", fileData);

    /* display the number in an ascii representation */
    gLogInfo << "Input:\n";
    for (int i = 0; i < eltCount; i++)
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
    gLogInfo << std::endl;

    /* initialize the inputs buffer */
    for (int i = 0; i < eltCount; i++)
        inputs[i] = 1.0 - float(fileData[i]) / 255.0;

    void* deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs.data(), memSize, cudaMemcpyHostToDevice));

    return deviceMem;
}

bool verifyOutput(int64_t eltCount, DataType dtype, void* buffer, int num)
{
    assert(elementSize(dtype) == sizeof(float));

    bool pass = false;

    size_t memSize = eltCount * elementSize(dtype);
    std::vector<float> outputs(eltCount);
    CHECK(cudaMemcpy(outputs.data(), buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = 0;
    for (int i = 0; i < eltCount; ++i)
        if (outputs[i] > outputs[maxIdx])
            maxIdx = i;

    std::ios::fmtflags prevSettings = gLogInfo.flags();
    gLogInfo.setf(std::ios::fixed, std::ios::floatfield);
    gLogInfo.precision(6);
    gLogInfo << "Output:\n";
    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        gLogInfo << eltIdx << " => " << setw(10) << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
        {
            gLogInfo << "***";
            if (eltIdx == num)
                pass = true;
        }
        gLogInfo << "\n";
    }
    gLogInfo.flags(prevSettings);

    gLogInfo << std::endl;

    return pass;
}

void loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                              IUffParser* parser, nvuffparser::IPluginFactoryExt* pluginFactory,
                              IHostMemory*& trtModelStream)
{
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    INetworkDefinition* network = builder->createNetwork();
    INetworkConfig* config = builder->createNetworkConfig();
    parser->setPluginFactoryExt(pluginFactory);

    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
    {
        gLogError << "Failure while parsing UFF file" << std::endl;
        return;
    }

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(MAX_WORKSPACE);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
    if (gArgs.runInFp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (gArgs.runInInt8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }
    samplesCommon::enableDLA(builder, config, gArgs.useDLACore);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        gLogError << "Unable to create engine" << std::endl;
        return;
    }

    /* we can clean the network and the parser */
    network->destroy();

    trtModelStream = engine->serialize();

    engine->destroy();
    builder->destroy();
    config->destroy();
    shutdownProtobufLibrary();
}

bool execute(ICudaEngine& engine)
{
    bool pass = true;

    IExecutionContext* context = engine.createExecutionContext();

    int batchSize = 1;

    int nbBindings = engine.getNbBindings();
    assert(nbBindings == 2);

    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (engine.bindingIsInput(i))
            bindingIdxInput = i;
        else
        {
            auto bufferSizesOutput = buffersSizes[i];
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first * elementSize(bufferSizesOutput.second));
        }
    }

    auto bufferSizesInput = buffersSizes[bindingIdxInput];

    int iterations = 1;
    int numberRun = 10;
    for (int i = 0; i < iterations; i++)
    {
        float total = 0, ms;
        for (int num = 0; num < numberRun; num++)
        {
            buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first,
                                                             bufferSizesInput.second, num);

            auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(batchSize, &buffers[0]);
            auto t_end = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            total += ms;

            for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
            {
                if (engine.bindingIsInput(bindingIdx))
                    continue;

                auto bufferSizesOutput = buffersSizes[bindingIdx];
                pass &= verifyOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                                     buffers[bindingIdx], num);
            }
            CHECK(cudaFree(buffers[bindingIdxInput]));
        }

        total /= numberRun;
        gLogInfo << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
    }

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();

    return pass;
}

struct PoolParameters
{
    /* Input dimensions */
    int mC, mH, mW;
    /* Output dimensions */
    int mP, mQ;
    /* Kernel size */
    int mR, mS;
    /* Stride */
    int mU, mV;
    /* Padding */
    int pH, pW;
    /* Pooling Function */
    PoolingType pType;
};

class UffPoolPlugin : public IPluginExt
{
public:
    /*Fully connected layer implementation. Bias weights is set to nullptr*/
    UffPoolPlugin(const nvuffparser::FieldCollection fc)
    {
        int nbFields = fc.nbFields;
        const nvuffparser::FieldMap* fields = fc.fields;
        std::string dFormat = "NCHW";
        mPoolingParams.pType = PoolingType::kMAX;
        // sample code parsing FieldCollection
        for (int i = 0; i < nbFields; ++i)
        {
            const char* name = fields[i].name;
            if (strcmp(name, "strides") == 0)
            {
                assert(fields[i].type == FieldType::kDIMS);
                const Dims sDims = *(static_cast<const Dims*>(fields[i].data));
                mPoolingParams.mU = sDims.d[1];
                mPoolingParams.mV = sDims.d[2];
            }
            else if (strcmp(name, "ksize") == 0)
            {
                assert(fields[i].type == FieldType::kDIMS);
                const Dims kDims = *(static_cast<const Dims*>(fields[i].data));
                mPoolingParams.mR = kDims.d[1];
                mPoolingParams.mS = kDims.d[2];
            }
            else if (strcmp(name, "padding") == 0)
            {
                assert(fields[i].type == FieldType::kCHAR);
                std::string padding(static_cast<const char*>(fields[i].data), fields[i].length);
            }
            else if (strcmp(name, "T") == 0)
            {
                assert(fields[i].type == FieldType::kDATATYPE);
                DataType dType = *(static_cast<const DataType*>(fields[i].data));
                (void) dType;
            }
            else if (strcmp(name, "data_format") == 0)
            {
                assert(fields[i].type == FieldType::kCHAR);
                std::string dFormat(static_cast<const char*>(fields[i].data), fields[i].length);
            }
        }

        mPoolingParams.pH = 0;
        mPoolingParams.pW = 0;
        mode = mPoolingParams.pType == PoolingType::kMAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    }

    /* create the plugin at runtime from a byte stream*/
    UffPoolPlugin(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        mPoolingParams.pType = static_cast<PoolingType>(read<int>(d));
        mPoolingParams.mC = read<int>(d);
        mPoolingParams.mH = read<int>(d);
        mPoolingParams.mW = read<int>(d);
        mPoolingParams.mP = read<int>(d);
        mPoolingParams.mQ = read<int>(d);
        mPoolingParams.mR = read<int>(d);
        mPoolingParams.mS = read<int>(d);
        mPoolingParams.mU = read<int>(d);
        mPoolingParams.mV = read<int>(d);
        mPoolingParams.pH = read<int>(d);
        mPoolingParams.pW = read<int>(d);
        mDataType = static_cast<DataType>(read<int>(d));

        mode = mPoolingParams.pType == PoolingType::kMAX ? CUDNN_POOLING_MAX : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

        assert(d == a + length);
    }

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
        DimsHW outDims((inputs[0].d[1] + mPoolingParams.pH * 2 - mPoolingParams.mR) / mPoolingParams.mU + 1,
                       (inputs[0].d[2] + mPoolingParams.pW * 2 - mPoolingParams.mS) / mPoolingParams.mV + 1);
        return Dims3(inputs[0].d[0], outDims.h(), outDims.w());
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW;
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
                             DataType type, PluginFormat format, int maxBatchSize) override
    {
        assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
        mPoolingParams.mC = inputDims[0].d[0];
        mPoolingParams.mH = inputDims[0].d[1];
        mPoolingParams.mW = inputDims[0].d[2];

        mPoolingParams.mP = outputDims[0].d[1];
        mPoolingParams.mQ = outputDims[0].d[2];

        mDataType = type;
    }
    int initialize() override
    {
        CHECK(cudnnCreate(&mCudnn));                         // initialize cudnn and cublas
        CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor)); // create cudnn tensor descriptors we need for bias addition
        CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
        CHECK(cudnnCreatePoolingDescriptor(&mPoolingDesc));
        CHECK(cudnnSetPooling2dDescriptor(mPoolingDesc, mode, CUDNN_NOT_PROPAGATE_NAN, mPoolingParams.mR, mPoolingParams.mS,
                                          mPoolingParams.pH, mPoolingParams.pW, mPoolingParams.mU, mPoolingParams.mV));

        return 0;
    }

    virtual void terminate() override
    {
        CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
        CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
        CHECK(cudnnDestroyPoolingDescriptor(mPoolingDesc));
        CHECK(cudnnDestroy(mCudnn));
    }

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        float kONE = 1.0f, kZERO = 0.0f;
        cudnnSetStream(mCudnn, stream);

        int N = 1;
        std::map<DataType, cudnnDataType_t> typeMap = {
            {DataType::kFLOAT, CUDNN_DATA_FLOAT},
            {DataType::kHALF, CUDNN_DATA_HALF}};
        assert(mDataType == DataType::kFLOAT || mDataType == DataType::kHALF);
        CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, typeMap[mDataType], N, mPoolingParams.mC, mPoolingParams.mH, mPoolingParams.mW));
        CHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, typeMap[mDataType], N, mPoolingParams.mC, mPoolingParams.mP, mPoolingParams.mQ));
        CHECK(cudnnPoolingForward(mCudnn, mPoolingDesc, &kONE, mSrcDescriptor, inputs[0], &kZERO, mDstDescriptor, outputs[0]));

        return 0;
    }

    virtual size_t getSerializationSize() override
    {
        return sizeof(int) * 13;
    }

    virtual void serialize(void* buffer) override
    {
        char *d = reinterpret_cast<char *>(buffer), *a = d;
        write(d, static_cast<int>(mPoolingParams.pType));
        write(d, mPoolingParams.mC);
        write(d, mPoolingParams.mH);
        write(d, mPoolingParams.mW);
        write(d, mPoolingParams.mP);
        write(d, mPoolingParams.mQ);
        write(d, mPoolingParams.mR);
        write(d, mPoolingParams.mS);
        write(d, mPoolingParams.mU);
        write(d, mPoolingParams.mV);
        write(d, mPoolingParams.pH);
        write(d, mPoolingParams.pW);
        write(d, static_cast<int>(mDataType));
        assert(d == a + getSerializationSize());
    }

private:
    template <typename T>
    void write(char*& buffer, const T& val)
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    cudnnHandle_t mCudnn;
    cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
    cudnnPoolingDescriptor_t mPoolingDesc;
    PoolParameters mPoolingParams;
    cudnnPoolingMode_t mode;
    DataType mDataType;
};

class PluginFactory : public nvinfer1::IPluginFactory, public nvuffparser::IPluginFactoryExt
{
public:
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights,
                                            int nbWeights, const nvuffparser::FieldCollection fc) override
    {
        try
        {
            assert(isPlugin(layerName));
            assert(mPlugin.get() == nullptr);
            mPlugin = std::unique_ptr<UffPoolPlugin>(new UffPoolPlugin(fc));
            return mPlugin.get();
        }
        catch (std::exception& e)
        {
            gLogError << e.what() << std::endl;
        }

        return nullptr;
    }

    /* deserialization plugin implementation */
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
        try
        {
            assert(isPlugin(layerName));
            //This plugin object is destroyed when engine is destroyed by calling
            //IPluginExt::destroy()
            mPlugin = std::unique_ptr<UffPoolPlugin>(new UffPoolPlugin(serialData, serialLength));
            return mPlugin.get();
        }
        catch (std::exception& e)
        {
            gLogError << e.what() << std::endl;
        }

        return nullptr;
    }

    bool isPlugin(const char* name) override
    {
        return isPluginExt(name);
    }

    bool isPluginExt(const char* name) override
    {
        /* Custom nodes have a '_' before actual name */
        return (name && name[0] == '_');
    }

    // User application destroys plugin when it is safe to do so.
    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
    void destroyPlugin()
    {
        mPlugin.reset();
    }

    std::unique_ptr<UffPoolPlugin> mPlugin{nullptr};
};

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_uff_plugin [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode.\n";
}

int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/mnist/", "data/mnist/"};
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    auto fileName = locateFile("lenet5_custom_pool.uff", gArgs.dataDirs);
    gLogInfo << fileName << std::endl;

    int maxBatchSize = 1;
    auto parser = createUffParser();

    /* Register tensorflow input */
    parser->registerInput("in", Dims3(1, 28, 28), UffInputOrder::kNCHW);
    parser->registerOutput("out");

    PluginFactory pluginFactorySerialize;
    IHostMemory* trtModelStream{nullptr};
    loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser, &pluginFactorySerialize, trtModelStream);
    assert(trtModelStream != nullptr);
    pluginFactorySerialize.destroyPlugin();
    parser->destroy();

    /* deserialize the engine */
    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }

    PluginFactory pluginFactory;
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), &pluginFactory);
    assert(engine != nullptr);
    trtModelStream->destroy();

    bool pass = execute(*engine);

    // Destroy the engine
    engine->destroy();
    runtime->destroy();

    // Destroy plugins created by factory
    pluginFactory.destroyPlugin();

    return gLogger.reportTest(sampleTest, pass);
}
