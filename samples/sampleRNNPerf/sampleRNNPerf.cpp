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

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <map>
#include <vector>

#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>

#include "NvInfer.h"
#include "logger.h"
#include "common.h"

using namespace nvinfer1;

const std::string gSampleName = "TensorRT.sample_rnn_perf";

static int gUseDLACore{-1};

enum class RNNTensors : int
{
    kIN = 0,
    kHIDDENIN = 1,
    kCELLIN = 2,
    kOUT = 3,
    kHIDDENOUT = 4,
    kCELLOUT = 5
};

struct RNN
{
    // RNN parameters
    RNNOperation operation{RNNOperation::kRELU};
    RNNDirection direction{RNNDirection::kUNIDIRECTION};
    RNNInputMode mode{RNNInputMode::kLINEAR};
    int nbLayers{2};
    int hiddenSize{512};

    // Data parameters
    int batchSize{1};
    int seqLen{1};
    int inputSize{512};

    // Execution parameters
    bool var{false};
    bool verbose{false};
    int workspace{25};
    int iterations{1};
    DataType type{DataType::kFLOAT};

    int dirMult() const { return direction == RNNDirection::kBIDIRECTION ? 2 : 1; }

    int inputDataSize() const { return batchSize * seqLen * (mode == RNNInputMode::kSKIP ? hiddenSize : inputSize); }

    int outputDataSize() const { return batchSize * seqLen * hiddenSize * dirMult(); }

    int hiddenDataSize() const { return batchSize * nbLayers * hiddenSize * dirMult(); }

    int inputHiddenDataSize() const { return hiddenDataSize(); }

    int outputHiddenDataSize() const { return hiddenDataSize(); }

    int inputCellDataSize() const { return hiddenDataSize(); }

    int outputCellDataSize() const { return hiddenDataSize(); }

    int weightSize(int layer, RNNGateType gate, bool isW) const;

    int biasSize(int, RNNGateType, bool) const { return hiddenSize; }

    void getGates(std::vector<RNNGateType>& gates) const;
};

int RNN::weightSize(int layer, RNNGateType, bool isW) const
{
    if (isW)
    {
        if (layer)
            return hiddenSize * hiddenSize * dirMult();
        else
            return mode == RNNInputMode::kSKIP ? 0 : hiddenSize * inputSize;
    }
    return hiddenSize * hiddenSize;
}

void RNN::getGates(std::vector<RNNGateType>& gates) const
{
    switch (operation)
    {
    case RNNOperation::kRELU:
    case RNNOperation::kTANH:
        gates.push_back(RNNGateType::kINPUT);
        break;
    case RNNOperation::kLSTM:
        gates.push_back(RNNGateType::kINPUT);
        gates.push_back(RNNGateType::kOUTPUT);
        gates.push_back(RNNGateType::kFORGET);
        gates.push_back(RNNGateType::kCELL);
        break;
    case RNNOperation::kGRU:
        gates.push_back(RNNGateType::kUPDATE);
        gates.push_back(RNNGateType::kRESET);
        gates.push_back(RNNGateType::kHIDDEN);
        break;
    default: assert(false);
    }
}

struct Profiler : public IProfiler
{
    std::map<std::string, float> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms) override
    {
        mProfile[layerName] += ms;
    }

    void printLayerTimes(int iterations)
    {
        for (auto& i : mProfile)
            gLogInfo << i.first << " " << i.second / iterations << "ms" << std::endl;
    }
};

template <typename T>
inline DataType dataTypeOfType()
{
    assert(false);
    return DataType(EnumMax<DataType>());
}

template <>
inline DataType dataTypeOfType<float>()
{
    return DataType::kFLOAT;
}

template <>
inline DataType dataTypeOfType<half>()
{
    return DataType::kHALF;
}

template <typename T>
inline void randomFill(std::vector<T>& data)
{
    std::generate(data.begin(), data.end(), [](){ return static_cast<T>(rand()); });
}

template <>
inline void randomFill<half>(std::vector<half>& data)
{
    std::generate(data.begin(), data.end(), [](){
        uint16_t value = static_cast<uint16_t>(rand() & 0xFFFF);
        return *reinterpret_cast<half*>(&value);
    });
}

const std::string tensorName(const std::vector<std::string>& tensors, RNNTensors t)
{
    size_t index = static_cast<size_t>(t);
    assert(index < tensors.size());
    return tensors[index];
}

template <typename T>
void setWeightsAndBiases(const RNN& rnn, IRNNv2Layer& rnnLayer, std::vector<std::vector<T>>& weightsAndBiases)
{
    std::vector<RNNGateType> gates;
    rnn.getGates(gates);
    for (int l = 0; l < rnn.nbLayers * rnn.dirMult(); ++l)
        for (auto g : gates)
            for (int r = 0; r < 2; ++r)
            {
                int weightSize = rnn.weightSize(l / rnn.dirMult(), g, r);
                if (weightSize)
                {
                    weightsAndBiases.emplace_back(weightSize);
                    std::vector<T>& weightsData = weightsAndBiases.back();
                    randomFill<T>(weightsData);
                    Weights weights{dataTypeOfType<T>(), weightsData.data(), weightSize};
                    rnnLayer.setWeightsForGate(l, g, r, weights);
                }

                int biasSize = rnn.biasSize(l / rnn.dirMult(), g, r);
                weightsAndBiases.emplace_back(biasSize);
                std::vector<T>& biasData = weightsAndBiases.back();
                randomFill<T>(biasData);
                Weights bias{dataTypeOfType<T>(), biasData.data(), biasSize};
                rnnLayer.setBiasForGate(l, g, r, bias);
            }
}

template <typename T>
ICudaEngine* createRNNEngine(const RNN& rnn, const std::vector<std::string>& tensors)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    INetworkConfig* config = builder->createNetworkConfig();
    assert(builder != nullptr);
    if (builder->platformHasFastFp16() && rnn.type == DataType::kHALF)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    // create the network
    INetworkDefinition* network = builder->createNetwork();

    auto dataIn = network->addInput(tensorName(tensors, RNNTensors::kIN).c_str(), rnn.type, Dims2{rnn.seqLen, rnn.inputSize});
    auto hiddenIn = network->addInput(tensorName(tensors, RNNTensors::kHIDDENIN).c_str(), rnn.type, Dims2{rnn.nbLayers, rnn.hiddenSize * rnn.dirMult()});

    auto rnnLayer = network->addRNNv2(*dataIn, rnn.nbLayers, rnn.hiddenSize, rnn.seqLen, rnn.operation);
    rnnLayer->setName("RNNv2");
    rnnLayer->setHiddenState(*hiddenIn);

    rnnLayer->getOutput(0)->setName(tensorName(tensors, RNNTensors::kOUT).c_str());
    network->markOutput(*rnnLayer->getOutput(0));
    rnnLayer->getOutput(0)->setType(rnn.type);

    rnnLayer->getOutput(1)->setName(tensorName(tensors, RNNTensors::kHIDDENOUT).c_str());
    network->markOutput(*rnnLayer->getOutput(1));
    rnnLayer->getOutput(1)->setType(rnn.type);

    if (rnn.operation == RNNOperation::kLSTM)
    {
        auto cellIn = network->addInput(tensorName(tensors, RNNTensors::kCELLIN).c_str(), rnn.type, Dims2{rnn.nbLayers, rnn.hiddenSize * rnn.dirMult()});
        rnnLayer->setCellState(*cellIn);
        rnnLayer->getOutput(2)->setName(tensorName(tensors, RNNTensors::kCELLOUT).c_str());
        network->markOutput(*rnnLayer->getOutput(2));
        rnnLayer->getOutput(2)->setType(rnn.type);
    }

    std::vector<std::vector<T>> weightsAndBiases;
    setWeightsAndBiases<T>(rnn, *rnnLayer, weightsAndBiases);

    // Add variable sequence lengths
    if (rnn.var)
    {
        Dims dims{0};
        auto seqLengths = network->addInput("sLen", DataType::kINT32, dims);
        seqLengths->setLocation(TensorLocation::kHOST);
        rnnLayer->setSequenceLengths(*seqLengths);
    }

    // Build the engine
    builder->setMaxBatchSize(rnn.batchSize);
    config->setMaxWorkspaceSize(1ULL << rnn.workspace);
    samplesCommon::enableDLA(builder, config, gUseDLACore);
    auto engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);

    network->destroy();
    builder->destroy();
    config->destroy();

    return engine;
}

template <typename T>
void fillBuffers(std::vector<void*>& buffers, const ICudaEngine& engine, const RNN& rnn, const std::vector<std::string>& tensors)
{
    std::map<std::string, int> sizes{
        {"in", rnn.inputDataSize()},
        {"hIn", rnn.inputHiddenDataSize()},
        {"cIn", rnn.inputCellDataSize()},
        {"out", rnn.outputDataSize()},
        {"hOut", rnn.outputHiddenDataSize()},
        {"cOut", rnn.outputCellDataSize()}};

    for (auto& tensor : tensors)
    {
        int index = engine.getBindingIndex(tensor.c_str());
        if (index == -1)
            continue;
        assert(index < engine.getNbBindings());
        std::vector<T> buffer(sizes[tensor]);
        randomFill<T>(buffer);
        CHECK(cudaMalloc(&buffers[index], sizes[tensor] * sizeof(T)));
        CHECK(cudaMemcpy(buffers[index], buffer.data(), sizes[tensor] * sizeof(T), cudaMemcpyHostToDevice));
    }
    int index = engine.getBindingIndex("sLen");
    if (index != -1)
    {
        int32_t* buffer = new int32_t[rnn.batchSize];
        for (int s = 0; s < rnn.batchSize; ++s)
            buffer[s] = 1 + rand() % rnn.seqLen;
        buffers[index] = buffer;
    }
}

void timeInference(const RNN& rnn, IExecutionContext& context, const std::vector<std::string>& tensors)
{
    const ICudaEngine& engine = context.getEngine();

    std::vector<void*> buffers(engine.getNbBindings(), nullptr);

    if (rnn.type == DataType::kFLOAT)
        fillBuffers<float>(buffers, engine, rnn, tensors);
    else
        fillBuffers<half>(buffers, engine, rnn, tensors);

    cudaProfilerStart();
    for (int i = 0; i < rnn.iterations; ++i)
        context.execute(rnn.batchSize, buffers.data());
    cudaProfilerStop();

    // Free host memory for sLen if used
    int index = engine.getBindingIndex("sLen");
    if (index != -1)
    {
        int32_t* buffer = static_cast<int32_t*>(buffers[index]);
        delete[] buffer;
        buffers[index] = nullptr;
    }
    for (auto buffer : buffers)
        if (buffer)
            CHECK(cudaFree(buffer));
}

void printHelp()
{
    std::cout << "./sample_rnn_perf: [-[h|b|s|o|l] <int>] [--useDLACore=<int>] [-[d|m]] [--help]\n"
              << "    -o <int>             Specify the operation (RELU=0, TANH=1, LSTM=2, GRU=3).\n"
              << "    -l <int>             Specify the number of nbLayers in the RNN.\n"
              << "    -h <int>             Specify the number of hidden weights.\n"
              << "    -b <int>             Specify the batch size.\n"
              << "    -s <int>             Specify the lengths of the sequences to process.\n"
              << "    -e <int>             Specify the data size or embedding dimension.\n"
              << "    -d                   Turn on bidirectional RNN.\n"
              << "    -k                   Turn on input skip input mode.\n"
              << "    -v                   Turn on variable sequence lengths.\n"
              << "    -f                   Turn on half precision.\n"
              << "    -i <int>             Specify the number of iterations to run the RNN.\n"
              << "    -w <int>             Specify the power of 2 to compute for temp workspace, defaults to 25 (32 MB).\n"
              << "    --useDLACore=<int>   Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.\n"
              << "    -V                   Turn on verbose logging messages."
              << std::endl;
}

bool parseArgs(int argc, const char** argv, RNN& rnn)
{
    for (int x = 1; x < argc; ++x)
    {
        std::string tmp(argv[x]);
        if (tmp == "-o" && (x + 1) < argc)
        {
            rnn.operation = RNNOperation(atoi(argv[++x]));
            if (int(rnn.operation) > EnumMax<RNNOperation>())
            {
                gLogError << "Invalid RNN operation" << std::endl;
                return false;
            }
        }
        else if (tmp == "-l" && (x + 1) < argc)
            rnn.nbLayers = atoi(argv[++x]);
        else if (tmp == "-h" && (x + 1) < argc)
            rnn.hiddenSize = atoi(argv[++x]);
        else if (tmp == "-b" && (x + 1) < argc)
            rnn.batchSize = atoi(argv[++x]);
        else if (tmp == "-s" && (x + 1) < argc)
            rnn.seqLen = atoi(argv[++x]);
        else if (tmp == "-e" && (x + 1) < argc)
            rnn.inputSize = atoi(argv[++x]);
        else if (tmp == "-d")
            rnn.direction = RNNDirection::kBIDIRECTION;
        else if (tmp == "-k")
            rnn.mode = RNNInputMode::kSKIP;
        else if (tmp == "-f")
            rnn.type = DataType::kHALF;
        else if (tmp == "-v")
            rnn.var = true;
        else if (tmp == "-i" && (x + 1) < argc)
            rnn.iterations = atoi(argv[++x]);
        else if (tmp == "-w" && (x + 1) < argc)
            rnn.workspace = atoi(argv[++x]);
        else if (tmp == "-V")
            rnn.verbose = true;
        else if (tmp.compare(0, 13, "--useDLACore=") == 0)
        {
            gUseDLACore = stoi(argv[x] + 13);
            rnn.type = DataType::kHALF;
        }
        else
        {
            gLogError << "Invalid argument" << std::endl;
            return false;
        }
    }
    if (static_cast<size_t>(rnn.workspace) >= sizeof(size_t) * CHAR_BIT)
    {
        gLogError << "Invalid workspace" << std::endl;
        return false;
    }

    return true;
}

int main(int argc, const char** argv)
{
    RNN rnn;
    Profiler profiler;

    std::vector<std::string> tensors{"in", "hIn", "cIn", "out", "hOut", "cOut"};

    bool argsOK = parseArgs(argc, argv, rnn);

    if (!argsOK)
    {
        printHelp();
        return EXIT_FAILURE;
    }
    if (rnn.verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    assert(rnn.type == DataType::kFLOAT || rnn.type == DataType::kHALF);
    ICudaEngine* engine = rnn.type == DataType::kFLOAT ? createRNNEngine<float>(rnn, tensors) : createRNNEngine<half>(rnn, tensors);

    IExecutionContext* context = engine->createExecutionContext();

    context->setProfiler(&profiler);

    timeInference(rnn, *context, tensors);
    profiler.printLayerTimes(rnn.iterations);

    context->destroy();
    engine->destroy();

    return gLogger.reportPass(sampleTest);
}
