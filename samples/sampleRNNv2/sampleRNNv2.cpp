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

//! This sample shows how to migrate to RNNv2 API from deprecated RNN API.

#include "NvInfer.h"
#include "NvUtils.h"
#include "logger.h"
#include "common.h"
#include "cuda_runtime_api.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

const std::string gSampleName = "TensorRT.sample_rnnv2";

namespace nvinfer1
{
enum ParamType
{
    Weight,
    Bias
};

enum WeightType
{
    Input,
    Recurrent,
};

enum NetworkType
{
    RNNv1,
    RNNv2
};

template <>
constexpr int EnumMax<NetworkType>()
{
    return 2;
}
}; // namespace nvinfer1

using namespace nvinfer1;

struct GateParams
{
    int layer;
    RNNGateType gateType;
    bool isW;
    Weights w;
};

int getDimsVolume(const Dims& dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

std::string operationToStr(RNNOperation op)
{
    switch (op)
    {
    case RNNOperation::kRELU: return "RELU";
    case RNNOperation::kTANH: return "TANH";
    case RNNOperation::kLSTM: return "LSTM";
    case RNNOperation::kGRU: return "GRU ";
    }

    return "InvalidOp";
}

std::string directionToStr(RNNDirection dir)
{
    if (dir == RNNDirection::kUNIDIRECTION)
        return "Uni";
    else
        return "Bi ";
}

// Gates of RNN operation in same order as in weights
vector<RNNGateType> operationToGates(RNNOperation op)
{
    vector<RNNGateType> gates;
    switch (op)
    {
    case RNNOperation::kRELU:
    {
        gates.push_back(RNNGateType::kINPUT);
    }
    break;
    case RNNOperation::kTANH:
    {
        gates.push_back(RNNGateType::kINPUT);
    }
    break;
    case RNNOperation::kLSTM:
    {
        gates.push_back(RNNGateType::kFORGET);
        gates.push_back(RNNGateType::kINPUT);
        gates.push_back(RNNGateType::kCELL);
        gates.push_back(RNNGateType::kOUTPUT);
    }
    break;
    case RNNOperation::kGRU:
    {
        gates.push_back(RNNGateType::kUPDATE);
        gates.push_back(RNNGateType::kRESET);
        gates.push_back(RNNGateType::kHIDDEN);
    }
    break;
    default:
        assert(false);
    }
    return gates;
}

void showHelp()
{
    std::cout << "This sample shows how to migrate to addRNNv2() from the deprecated addRNN() API.\n\n"
                 "It demonstrates how to:\n"
                 "- Replace addRNN() with addRNNv2(). See addRNNv2Layer() in sample code.\n"
                 "- Convert old weights to per-layer per-gate weights for RNNv2. See convertRNNv1To2Params() in sample code.\n"
                 "- Set converted weights to RNNv2 layer. See setRNNv2Params() in sample code.\n";
}

class SampleRNNv2
{
public:
    SampleRNNv2(RNNOperation op, RNNDirection direction, int layerNum)
        : op_(op)
        , direction_(direction)
        , layerNum_(layerNum)
    {
        inputSize_ = 4;
        hiddenSize_ = 8;
        seqLen_ = 2;
        maxBatchSize_ = 1;
        inputMode_ = RNNInputMode::kLINEAR;
        outputTensorName_ = "OutputTensor";
        bufferSeed_ = rand();
        weightSeed_ = rand();
        biasSeed_ = rand();
    }

    ~SampleRNNv2()
    {
        // Free buffers
        for (void* devBuf : buffers_)
        {
            cudaFree(devBuf);
        }

        // Destroy engines
        for (ICudaEngine* engine : engines_)
        {
            if (engine)
                engine->destroy();
        }
    }

    // Build two networks: one with RNN layers, the second with RNNv2 layers.
    // Set weights in RNNv2 by migrating it from old format.
    // Execute both the networks and compare their results for sanity check.
    bool run()
    {
        makeRNNParams();
        createNetwork(RNNv1);
        createNetwork(RNNv2);
        makeIOBuffers(maxBatchSize_);

        for (ICudaEngine* engine : engines_)
        {
            IExecutionContext* context = engine->createExecutionContext();
            setInputBuffer(engine, maxBatchSize_);
            context->execute(maxBatchSize_, buffers_.data());
            copyOutputBuffer(engine, maxBatchSize_);
        }

        return compareOutputBuffers();
    }

private:
    bool compareOutputBuffers()
    {
        if (outBuffers_.at(0) == outBuffers_.at(1))
        {
            gLogInfo << "Outputs match" << std::endl;
            return true;
        }
        gLogInfo << "Outputs do not match!" << std::endl;
        return false;
    }

    void addRNNv1Layer(INetworkDefinition* network, ITensor* inputTensor)
    {
        Weights w;
        Weights b;
        getRNNv1Params(w, b);

        IRNNLayer* layer = network->addRNN(
            *inputTensor,
            layerNum_,
            (size_t) hiddenSize_,
            seqLen_,
            op_,
            inputMode_,
            direction_,
            w,
            b);
        assert(layer);

        layer->setName("RNNv1Layer");
        layer->getOutput(0)->setName(outputTensorName_.c_str());
        network->markOutput(*layer->getOutput(0));
    }

    // This method demonstrates how to use RNNv2 API.
    void addRNNv2Layer(INetworkDefinition* network, ITensor* inputTensor)
    {
        // Call to new addRNNv2 API
        IRNNv2Layer* layer = network->addRNNv2(
            *inputTensor,
            layerNum_,
            hiddenSize_,
            seqLen_,
            op_);
        assert(layer);

        layer->setInputMode(inputMode_);
        layer->setDirection(direction_);

        // Weights in old format
        Weights w;
        Weights b;
        getRNNv1Params(w, b);

        // Convert to weights suitable for v2 API
        vector<GateParams> wParams;
        vector<GateParams> bParams;
        convertRNNv1To2Params(w, b, wParams, bParams);

        // Set weights for v2 layer
        setRNNv2Params(layer, wParams, ParamType::Weight);
        setRNNv2Params(layer, bParams, ParamType::Bias);

        layer->setName("RNNv2Layer");
        layer->getOutput(0)->setName(outputTensorName_.c_str());
        network->markOutput(*layer->getOutput(0));
    }

    void setRNNv2Params(IRNNv2Layer* layer, const vector<GateParams>& gParamsVec, ParamType pType)
    {
        for (const GateParams& gParams : gParamsVec)
        {
            if (pType == ParamType::Weight)
            {
                layer->setWeightsForGate(gParams.layer, gParams.gateType, gParams.isW, gParams.w);
            }
            else
            {
                layer->setBiasForGate(gParams.layer, gParams.gateType, gParams.isW, gParams.w);
            }
        }
    }

    void createNetwork(NetworkType netType)
    {
        IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
        assert(builder != nullptr);
        INetworkConfig* config = builder->createNetworkConfig();
        INetworkDefinition* network = builder->createNetwork();

        const Dims inDims = (netType == NetworkType::RNNv1) ? Dims3{seqLen_, maxBatchSize_, inputSize_} : Dims3{maxBatchSize_, seqLen_, inputSize_};
        ITensor* inputTensor = network->addInput("InputTensor", DataType::kFLOAT, inDims);
        assert(inputTensor);

        if (netType == NetworkType::RNNv1)
        {
            addRNNv1Layer(network, inputTensor);
        }
        else
        {
            addRNNv2Layer(network, inputTensor);
        }

        builder->setMaxBatchSize(maxBatchSize_);
        config->setMaxWorkspaceSize(1_GB);

        ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
        assert(engine);
        engines_.push_back(engine);

        network->destroy();
        builder->destroy();
        config->destroy();
    }

    // Allocate device memory for buffers that are passed to engine
    // We reuse the same buffers for all engines.
    void makeIOBuffers(int batchSize)
    {
        assert(buffers_.empty());

        const int bufferNum = engines_[0]->getNbBindings();

        for (int i = 0; i < bufferNum; ++i)
        {
            const Dims dims = engines_[0]->getBindingDimensions(i);
            const int vol = batchSize * getDimsVolume(dims);
            void* dPtr = nullptr;
            cudaMalloc(&dPtr, vol * sizeof(float));
            buffers_.push_back(dPtr);
        }
    }

    void setInputBuffer(ICudaEngine* engine, int batchSize)
    {
        const int bufferNum = engine->getNbBindings();

        for (int i = 0; i < bufferNum; ++i)
        {
            // Skip output binding of network
            if (!engine->bindingIsInput(i))
                continue;

            const Dims dims = engine->getBindingDimensions(i);
            const int vol = batchSize * getDimsVolume(dims);
            vector<float> inBuf(vol);
            randomizeBuffer(inBuf.data(), vol, bufferSeed_); // Generate same random input every time
            cudaMemcpy(buffers_[i], inBuf.data(), sizeof(float) * vol, cudaMemcpyHostToDevice);
        }
    }

    void copyOutputBuffer(ICudaEngine* engine, int batchSize)
    {
        const int bufferNum = engine->getNbBindings();

        for (int i = 0; i < bufferNum; ++i)
        {
            // Skip input binding of network
            if (engine->bindingIsInput(i))
                continue;

            const Dims dims = engine->getBindingDimensions(i);
            const int vol = batchSize * getDimsVolume(dims);
            vector<float> outBuffer(vol);
            cudaMemcpy(outBuffer.data(), buffers_[i], sizeof(float) * vol, cudaMemcpyDeviceToHost);
            outBuffers_.push_back(outBuffer);
        }
    }

    void makeRNNParams()
    {
        const vector<RNNGateType> gates = operationToGates(op_);
        const int gateNum = gates.size();
        const int dirMult = (direction_ == RNNDirection::kUNIDIRECTION) ? 1 : 2;

        const int wNum = (inputSize_ * hiddenSize_ * gateNum * dirMult) +                // Layer 1
            (hiddenSize_ * hiddenSize_ * gateNum * dirMult * dirMult * (layerNum_ - 1)); // Layer 2...N
        const int rNum = hiddenSize_ * hiddenSize_ * gateNum * dirMult * layerNum_;
        weightBuffer_.resize(wNum + rNum);
        randomizeBuffer(weightBuffer_.data(), wNum + rNum, weightSeed_);

        const int bNum = layerNum_ * hiddenSize_ * gateNum * 2 * dirMult;
        biasBuffer_.resize(bNum);
        randomizeBuffer(biasBuffer_.data(), bNum, biasSeed_);
    }

    void getRNNv1Params(Weights& w, Weights& b)
    {
        w.type = DataType::kFLOAT;
        w.count = weightBuffer_.size();
        w.values = weightBuffer_.data();

        b.type = DataType::kFLOAT;
        b.count = biasBuffer_.size();
        b.values = biasBuffer_.data();
    }

    Weights getWeightsAtOffset(const Weights& inW, int offset, int wnum)
    {
        Weights w;
        w.type = inW.type;
        w.count = wnum;
        w.values = (void*) ((float*) inW.values + offset);
        return w;
    }

    // Convert weights to per-layer per-gate weights required by RNNv2 API
    void convertRNNv1To2Params(
        const Weights& inW,
        const Weights& inB,
        vector<GateParams>& wParams,
        vector<GateParams>& bParams)
    {
        assert(wParams.empty());
        assert(bParams.empty());

        vector<RNNGateType> gates = operationToGates(op_);

        const int dirMult = (direction_ == RNNDirection::kUNIDIRECTION) ? 1 : 2;
        int wOffset = 0;
        int bOffset = 0;

        // Iterate each RNN layer
        for (int li = 0; li < layerNum_; ++li)
        {
            const int wCols = (li == 0) ? inputSize_ : (hiddenSize_ * dirMult);

            // Uni or bi-dir layers in each RNN layer
            for (int di = 0; di < dirMult; ++di)
            {
                const int v2Layer = (li * dirMult) + di;

                // Iterate W and R params
                for (int i = 0; i < 2; ++i)
                {
                    const bool isW = (i == 0);
                    const int wNum = (isW ? wCols : hiddenSize_) * hiddenSize_;
                    const int bNum = hiddenSize_;

                    for (RNNGateType g : gates)
                    {
                        {
                            const Weights w = getWeightsAtOffset(inW, wOffset, wNum);
                            const GateParams wp{v2Layer, g, isW, w};
                            wParams.push_back(wp);
                        }

                        {
                            const Weights b = getWeightsAtOffset(inB, bOffset, bNum);
                            const GateParams bp{v2Layer, g, isW, b};
                            bParams.push_back(bp);
                        }

                        wOffset += wNum;
                        bOffset += bNum;
                    }
                }
            }
        }
    }

    void randomizeBuffer(void* ptr, int num, int seed)
    {
        srand(seed);
        float* fPtr = static_cast<float*>(ptr);
        for (int i = 0; i < num; ++i)
        {
            fPtr[i] = (float) rand() / (rand() + 1);
        }
    }

    vector<ICudaEngine*> engines_;
    vector<void*> buffers_;
    RNNOperation op_;
    RNNInputMode inputMode_;
    RNNDirection direction_;
    int layerNum_;
    int inputSize_;
    int hiddenSize_;
    int seqLen_;
    int maxBatchSize_;
    std::string outputTensorName_;
    vector<float> weightBuffer_;
    vector<float> biasBuffer_;
    vector<vector<float>> outBuffers_;
    int bufferSeed_;
    int weightSeed_;
    int biasSeed_;
};

int main(int argc, char** argv)
{
    // Show user help
    if (argc > 2)
    {
        showHelp();
        return EXIT_FAILURE;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);


    bool pass = true;

    // Try RNN networks with all types of operations, directions and layer numbers
    for (int op = 0; op < EnumMax<RNNOperation>(); ++op)
    {
        const RNNOperation opType = static_cast<RNNOperation>(op);

        for (int dir = 0; dir < EnumMax<RNNDirection>(); ++dir)
        {
            const RNNDirection dirType = static_cast<RNNDirection>(dir);

            for (int layerNum = 1; layerNum <= 8; layerNum *= 2)
            {
                gLogInfo << "RNN Op: " << operationToStr(opType) << " Dir: " << directionToStr(dirType) << " Layers: " << layerNum << std::endl;
                SampleRNNv2 sample(opType, dirType, layerNum);
                pass &= sample.run();
            }
        }
    }

    return gLogger.reportTest(sampleTest, pass);
}
