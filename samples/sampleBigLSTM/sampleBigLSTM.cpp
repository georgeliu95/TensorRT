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
#include "cuda_runtime_api.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <cuda_profiler_api.h>

#include "data_utils.h"
#include "error_util.h"
#include "bind_data.h"
#include "weight_utils.h"
#include "params.h"
#include "profiler.h"
#include "type_utils.h"
#include "logger.h"
#include "common.h"

#define COPY2D
//#define INIT_HIDDEN_IN

const std::string gSampleName = "TensorRT.sample_big_lstm";

// When IsDebug is true, the Softmax layer
// and perplexity calculation are disabled
const bool IsDebug = false; //true;
using namespace RNNDataUtil;

// Information describing the network
static const int OUTPUT_SIZE = 793470;

const char* INPUT_BLOB_NAME = "data";
const char* HIDDEN_IN_BLOB_NAME = "hiddenIn";
const char* CELL_IN_BLOB_NAME = "cellIn";
const char* OUTPUT_BLOB_NAME = "prob";
const char* HIDDEN_OUT_BLOB_NAME = "hiddenOut";
const char* LSTM_OUT_BLOB_NAME = "lstmOut";
const char* CELL_OUT_BLOB_NAME = "cellOut";
const char* PROJECT_IN_BLOB_NAME = "projIn";
const char* PROJECT_OUT_BLOB_NAME = "projOut";
const char* LOGITS_IN_BLOB_NAME = "logitsIn";

using namespace nvinfer1;

Params gParams;

std::string appendNumToString(std::string str, uint32_t z)
{
    std::stringstream ss;
    ss << str << z;
    return ss.str();
}

template <nvinfer1::DataType nvInferType>
ICudaEngine* APIToModelLSTM(std::map<std::string, Weights>& weightMap, uint32_t z)
{
    typedef typename InferTypeToValue<nvInferType>::value_type value_type;
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());

    // create the model to populate the network, then set the outputs and create an engine
    INetworkDefinition* network = builder->createNetwork();
    INetworkConfig* config = builder->createNetworkConfig();

    auto data = network->addInput(appendNumToString(INPUT_BLOB_NAME, z).c_str(), nvInferType, Dims3{1, (int) gParams.minibatchSize, int(gParams.dataSize + gParams.projectedSize)});
    assert(data != nullptr);

    auto cellIn = network->addInput(appendNumToString(CELL_IN_BLOB_NAME, z).c_str(), nvInferType, Dims3{1, (int) gParams.minibatchSize, (int) gParams.cellSize});
    assert(cellIn != nullptr);

#ifdef INIT_HIDDEN_IN
    auto hiddenIn = network->addInput(appendNumToString(HIDDEN_IN_BLOB_NAME, z).c_str(), nvInferType, Dims3{1, (int) gParams.minibatchSize, (int) gParams.cellSize});
    assert(hiddenIn != nullptr);
#endif

    auto rnnwts = weightMap["rnnweight2"];
    auto rnnbias = weightMap["rnnbias2"];

    // split weights by layer
    Weights rnnwtL, rnnbiasL;

    rnnwtL.count = (gParams.cellSize * 4 * (gParams.dataSize + gParams.projectedSize) + 4 * gParams.cellSize * gParams.cellSize); // 4*gParams.cellSize*gParams.cellSize - U weights set to 0
    rnnwtL.values = (void*) ((value_type*) rnnwts.values + z * rnnwtL.count);
    rnnwtL.type = nvInferType;

    rnnbiasL.count = (8 * gParams.cellSize);
    rnnbiasL.values = (void*) ((value_type*) rnnbias.values + z * rnnbiasL.count);
    rnnbiasL.type = nvInferType;

    IRNNv2Layer* rnn = network->addRNNv2(*data, 1, gParams.cellSize, 1, RNNOperation::kLSTM);
    assert(rnn != nullptr);
    rnn->setInputMode(RNNInputMode::kLINEAR);
    rnn->setDirection(RNNDirection::kUNIDIRECTION);

    setRNNLayerWeights<nvInferType>(
        rnn,
        rnnwtL.values,
        gParams.dataSize,
        gParams.projectedSize,
        gParams.cellSize,
        0);
    setRNNLayerBiases<nvInferType>(
        rnn,
        rnnbiasL.values,
        gParams.cellSize,
        0);

#ifdef INIT_HIDDEN_IN
    rnn->setHiddenState(*hiddenIn);
#endif
    if (rnn->getOperation() == RNNOperation::kLSTM)
        rnn->setCellState(*cellIn);

    rnn->getOutput(0)->setName(appendNumToString(LSTM_OUT_BLOB_NAME, z).c_str());
    network->markOutput(*rnn->getOutput(0));

    if (rnn->getOperation() == RNNOperation::kLSTM)
    {
        rnn->getOutput(2)->setName(appendNumToString(CELL_OUT_BLOB_NAME, z).c_str());
        network->markOutput(*rnn->getOutput(2));
    }
    rnn->getOutput(0)->setType(nvInferType);
    rnn->getOutput(2)->setType(nvInferType);

    // Build the engine
    builder->setMaxBatchSize(gParams.minibatchSize);
    config->setMaxWorkspaceSize(2_GB);
    if (gParams.half2)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (gParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }
    samplesCommon::setDummyInt8Scales(config, network);

    auto engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);
    // we don't need the network any more
    network->destroy();
    builder->destroy();
    config->destroy();
    return engine;
}

template <nvinfer1::DataType nvInferType>
ICudaEngine* APIToModelPROJECT(std::map<std::string, Weights>& weightMap, uint32_t z)
{
    typedef typename InferTypeToValue<nvInferType>::value_type value_type;
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());

    // create the model to populate the network, then set the outputs and create an engine
    INetworkDefinition* network = builder->createNetwork();
    INetworkConfig* config = builder->createNetworkConfig();

    auto data = network->addInput(appendNumToString(PROJECT_IN_BLOB_NAME, z).c_str(), nvInferType, Dims3{(int) gParams.cellSize, 1, 1});
    assert(data != nullptr);

    auto rnnwtsp = weightMap["rnnweightp2"];

    // Projection layer does not need bias, so set it to 0
    Weights rnnbiasp;
    rnnbiasp.type = nvInferType;
    rnnbiasp.values = nullptr;
    rnnbiasp.count = 0;

    /*    rnnbiasp.count = gParams.projectedSize;
    uint32_t sizeBytes = (nvInferType == DataType::kFLOAT)? 4 * gParams.projectedSize : 2 * gParams.projectedSize;
    rnnbiasp.values = malloc(sizeBytes);
    memset((void* )rnnbiasp.values, 0, sizeBytes);*/

    // split weights by layer
    Weights rnnwtpL;

    rnnwtpL.count = gParams.projectedSize * gParams.cellSize;
    rnnwtpL.values = (void*) ((value_type*) rnnwtsp.values + z * rnnwtpL.count);
    rnnwtpL.type = nvInferType;

    // add projection layer
    IFullyConnectedLayer* proj = network->addFullyConnected(*data, gParams.projectedSize, rnnwtpL, rnnbiasp);
    assert(proj != nullptr);

    proj->getOutput(0)->setName(appendNumToString(PROJECT_OUT_BLOB_NAME, z).c_str());
    network->markOutput(*proj->getOutput(0));
    proj->getOutput(0)->setType(nvInferType);

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    config->setMaxWorkspaceSize(512_MB);
    if (gParams.half2)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    auto engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);
    // we don't need the network any more
    network->destroy();
    builder->destroy();
    config->destroy();
    /*    free((void*)rnnbiasp.values);*/
    return engine;
}

template <nvinfer1::DataType nvInferType>
ICudaEngine* APIToModelLOGITS(std::map<std::string, Weights>& weightMap)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());

    // create the model to populate the network, then set the outputs and create an engine
    INetworkDefinition* network = builder->createNetwork();
    INetworkConfig* config = builder->createNetworkConfig();

    auto input = network->addInput(LOGITS_IN_BLOB_NAME, nvInferType, Dims3{(int) gParams.projectedSize, 1, 1});
    assert(input != nullptr);

    auto tffcwts = weightMap["rnnfcw"];
    auto bias = weightMap["rnnfcb"];
    auto fc = network->addFullyConnected(*input, OUTPUT_SIZE, tffcwts, bias);
    assert(fc != nullptr);

    if (gParams.sm && !IsDebug)
    {
        fc->getOutput(0)->setName("FC output");
        // Add a softmax layer to determine the probability.
        auto prob = network->addSoftMax(*fc->getOutput(0));
        assert(prob != nullptr);
        prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
        network->markOutput(*prob->getOutput(0));
        prob->getOutput(0)->setType(nvInferType);
    }
    else
    {
        fc->getOutput(0)->setName(OUTPUT_BLOB_NAME);
        network->markOutput(*fc->getOutput(0));
        fc->getOutput(0)->setType(nvInferType);
    }

    // Build the engine
    if (gParams.perplexity || IsDebug)
        builder->setMaxBatchSize(gParams.batchSize * gParams.seqSize);
    else
        builder->setMaxBatchSize(gParams.batchSize);
    config->setMaxWorkspaceSize(128_MB);
    if (gParams.half2)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    auto engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine != nullptr);
    // we don't need the network any more
    network->destroy();
    builder->destroy();
    config->destroy();
    return engine;
}

template <class value_type>
void stepOnce(BindData<value_type>** lstm, BindData<value_type>** project, BindData<value_type>& logits,
              cudaStream_t& stream, IExecutionContext** contextLSTM, IExecutionContext** contextPROJECT,
              IExecutionContext* contextLOGITS, value_type* stateBuffer, float* avgTime)
{
    uint32_t lastL = gParams.layers - 1;
    uint32_t lastT = gParams.seqSize - 1;

    const uint32_t iCellId = 1;
#ifdef INIT_HIDDEN_IN
    const uint32_t iCellId = 2;
#endif

    if (IsDebug)
        gParams.perplexity = false;
    // store first layer's beginning of the input position (input data for all timesptes is stored there)
    value_type* startInputPtr = *lstm[0]->getInputPtr(0);

    std::vector<cudaStream_t> layerStream(gParams.layers);
    std::vector<cudaStream_t> copyStream(gParams.layers - 1);
    for (uint32_t z = 0; z < gParams.layers; z++)
    {
        CHECK(cudaStreamCreate(&layerStream[z]));
        if (z < lastL)
        {
            CHECK(cudaStreamCreate(&copyStream[z]));
        }
    }

    std::vector<std::vector<cudaEvent_t>> layerEvent(gParams.seqSize);
    std::vector<std::vector<cudaEvent_t>> copyEvent(gParams.seqSize);

    for (uint32_t i = 0; i < gParams.seqSize; i++)
    {
        layerEvent[i] = (std::vector<cudaEvent_t>(gParams.layers));
        copyEvent[i] = (std::vector<cudaEvent_t>(gParams.layers));
    }
    for (uint32_t t = 0; t < gParams.seqSize; t++)
        for (uint32_t z = 0; z < gParams.layers; z++)
        {
            CHECK(cudaEventCreateWithFlags(&layerEvent[t][z], cudaEventDisableTiming));
            CHECK(cudaEventCreateWithFlags(&copyEvent[t][z], cudaEventDisableTiming));
        }

    // Execute asynchronously
    cudaStreamSynchronize(stream);

    cudaEvent_t start1, start2, end;
    CHECK(cudaEventCreateWithFlags(&start1, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&start2, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    for (int i = 0; i < 3; i++)
        avgTime[i] = 0.f;
    cudaProfilerStart();

    value_type* cellBuffer = stateBuffer;
    value_type* hiddenBuffer = stateBuffer + gParams.layers * gParams.batchSize * gParams.cellSize;
    for (int r = 0; r < gParams.avgRuns; r++)
    {
        // get the state from the previous iteration:
        for (uint32_t z = 0; z < gParams.layers; z++)
        {
            CHECK(cudaMemcpyAsync(*lstm[z]->getInputPtr(iCellId),
                                  cellBuffer + z * gParams.batchSize * gParams.cellSize,
                                  gParams.cellSize * sizeof(value_type) * gParams.batchSize,
                                  cudaMemcpyDeviceToDevice, layerStream[z]));

#ifdef COPY2D
            CHECK(cudaMemcpy2DAsync(*lstm[z]->getInputPtr(0) + gParams.dataSize,
                                    (gParams.dataSize + gParams.projectedSize) * sizeof(value_type),
                                    hiddenBuffer + z * gParams.projectedSize * gParams.batchSize,
                                    gParams.projectedSize * sizeof(value_type),
                                    gParams.projectedSize * sizeof(value_type), gParams.batchSize,
                                    cudaMemcpyDeviceToDevice, layerStream[z]));
#else
            for (uint32_t n = 0; n < gParams.batchSize; n++)
                CHECK(cudaMemcpyAsync(*lstm[z]->getInputPtr(0) + gParams.dataSize
                                          + n * (gParams.dataSize + gParams.projectedSize),
                                      hiddenBuffer + z * gParams.projectedSize * gParams.batchSize + n * gParams.projectedSize,
                                      gParams.projectedSize * sizeof(value_type),
                                      cudaMemcpyDeviceToDevice, layerStream[z]));
#endif
        }

        CHECK(cudaEventRecord(start1, stream));

        for (uint32_t t = 0; t < gParams.seqSize; t++)
        {
            for (uint32_t z = 0; z < gParams.layers; z++)
            {
                // check data arrived from previous layer:
                if (z > 0)
                {
                    CHECK(cudaStreamWaitEvent(layerStream[z], copyEvent[t][z - 1], 0));
                }
                if (t > 0)
                {
                    CHECK(cudaStreamWaitEvent(layerStream[z], copyEvent[t - 1][z], 0));
                }
                if ((gParams.perplexity || IsDebug) && z == lastL && t > 0)
                {
#ifdef COPY2D
                    CHECK(cudaMemcpyAsync(*(logits.getInputPtr(0)) + gParams.batchSize * (t - 1) * gParams.projectedSize,
                                          *(project[z]->getOutputPtr(0)), gParams.projectedSize * sizeof(value_type) * gParams.batchSize,
                                          cudaMemcpyDeviceToDevice, layerStream[z]));
#else
                    for (uint32_t n = 0; n < gParams.batchSize; n++)
                        CHECK(cudaMemcpyAsync(*(logits.getInputPtr(0)) + gParams.batchSize * (t - 1) * gParams.projectedSize + n * gParams.projectedSize,
                                              *(project[z]->getOutputPtr(0)) + n * gParams.projectedSize,
                                              gParams.projectedSize * sizeof(value_type), cudaMemcpyDeviceToDevice, layerStream[z]));
#endif
                }
                contextLSTM[z]->enqueue(1, lstm[z]->buffers(), layerStream[z], nullptr);
                std::swap(*project[z]->getInputPtr(0), *lstm[z]->getOutputPtr(0));
                /*                cudaMemcpyAsync(*project[z]->getInputPtr(0),
                                *lstm[z]->getOutputPtr(0),
                                gParams.batchSize * gParams.cellSize * sizeof(value_type),
                                cudaMemcpyDeviceToDevice, layerStream[z]);*/
                contextPROJECT[z]->enqueue(gParams.batchSize, project[z]->buffers(), layerStream[z], nullptr);
                CHECK(cudaEventRecord(layerEvent[t][z], layerStream[z]));

                if (z < lastL)
                {
                    if (t > 0)
                        CHECK(cudaStreamWaitEvent(copyStream[z], layerEvent[t - 1][z + 1], 0));
                    CHECK(cudaStreamWaitEvent(copyStream[z], layerEvent[t][z], 0));
#ifdef COPY2D
                    CHECK(cudaMemcpy2DAsync(*(lstm[z + 1]->getInputPtr(0)),
                                            (gParams.dataSize + gParams.projectedSize) * sizeof(value_type),
                                            *(project[z]->getOutputPtr(0)), gParams.projectedSize * sizeof(value_type),
                                            gParams.projectedSize * sizeof(value_type), gParams.batchSize,
                                            cudaMemcpyDeviceToDevice, copyStream[z]));
#else
                    for (uint32_t n = 0; n < gParams.batchSize; n++)
                        CHECK(cudaMemcpyAsync(*(lstm[z + 1]->getInputPtr(0))
                                                  + n * (gParams.dataSize + gParams.projectedSize),
                                              *(project[z]->getOutputPtr(0))
                                                  + n * gParams.projectedSize,
                                              gParams.projectedSize * sizeof(value_type),
                                              cudaMemcpyDeviceToDevice, copyStream[z]));
#endif
                    CHECK(cudaEventRecord(copyEvent[t][z], copyStream[z]));
                }

                if (t < lastT)
                {
                    // swapping CELL state
                    std::swap(*lstm[z]->getOutputPtr(1), *lstm[z]->getInputPtr(iCellId));
                    if (z == 0)
                        *lstm[0]->getInputPtr(0) = *lstm[0]->getInputPtr(0) + gParams.batchSize * (gParams.dataSize + gParams.projectedSize);
                        // copy hidden state from previous time step
#ifdef COPY2D
                    CHECK(cudaMemcpy2DAsync(*lstm[z]->getInputPtr(0) + gParams.dataSize,
                                            (gParams.dataSize + gParams.projectedSize) * sizeof(value_type),
                                            *project[z]->getOutputPtr(0),
                                            gParams.projectedSize * sizeof(value_type),
                                            gParams.projectedSize * sizeof(value_type), gParams.batchSize,
                                            cudaMemcpyDeviceToDevice, layerStream[z]));
#else
                    for (uint32_t n = 0; n < gParams.batchSize; n++)
                        CHECK(cudaMemcpyAsync(*lstm[z]->getInputPtr(0) + gParams.dataSize
                                                  + n * (gParams.dataSize + gParams.projectedSize),
                                              *project[z]->getOutputPtr(0)
                                                  + n * gParams.projectedSize,
                                              gParams.projectedSize * sizeof(value_type), cudaMemcpyDeviceToDevice, layerStream[z]));
#endif
                }
                if (z == lastL && t == lastT)
                {
                    // pass LSTMP output to the output layer
                    CHECK(cudaStreamWaitEvent(stream, layerEvent[t][z], 0));
                    CHECK(cudaEventRecord(start2, stream));
                    size_t stride = (gParams.perplexity || IsDebug) ? lastT : 0;
#ifdef COPY2D
                    CHECK(cudaMemcpyAsync(*(logits.getInputPtr(0)) + gParams.batchSize * stride * gParams.projectedSize,
                                          *(project[z]->getOutputPtr(0)),
                                          gParams.projectedSize * sizeof(value_type) * gParams.batchSize,
                                          cudaMemcpyDeviceToDevice, layerStream[z]));
#else
                    for (uint32_t n = 0; n < gParams.batchSize; n++)
                        CHECK(cudaMemcpyAsync(*(logits.getInputPtr(0)) + gParams.batchSize * stride * gParams.projectedSize + n * gParams.projectedSize,
                                              *(project[z]->getOutputPtr(0)) + n * gParams.projectedSize,
                                              gParams.projectedSize * sizeof(value_type), cudaMemcpyDeviceToDevice, layerStream[z]));
#endif
                    //std::swap(*lstmp[lastL]->getOutputPtr(0), *logits.getInputPtr(0));
                }
            }
        }

        /*   std::swap(lstmp.getOutputRef(0), logits.getInputRef(0));*/
        if (gParams.perplexity || IsDebug)
            contextLOGITS->enqueue(gParams.batchSize * gParams.seqSize, logits.buffers(), layerStream[lastL], nullptr);
        else
            contextLOGITS->enqueue(gParams.batchSize, logits.buffers(), layerStream[lastL], nullptr);

        cudaStreamSynchronize(layerStream[lastL]);
        CHECK(cudaEventRecord(end, stream));
        CHECK(cudaEventSynchronize(end));

        float ms[2] = {0.f};
        CHECK(cudaEventElapsedTime(&ms[0], start1, start2));
        CHECK(cudaEventElapsedTime(&ms[1], start2, end));
        avgTime[0] += ms[0];
        avgTime[1] += ms[1];
        avgTime[2] += (ms[0] + ms[1]);

        if (r == gParams.avgRuns - 1)
        {
            // store output cell states in a buffer for the next iteration
            for (uint32_t z = 0; z < gParams.layers; z++)
            {
                CHECK(cudaMemcpyAsync(cellBuffer + z * gParams.batchSize * gParams.cellSize,
                                      *lstm[z]->getOutputPtr(1),
                                      gParams.cellSize * sizeof(value_type) * gParams.batchSize,
                                      cudaMemcpyDeviceToDevice, layerStream[lastL]));

                CHECK(cudaMemcpyAsync(hiddenBuffer + z * gParams.projectedSize * gParams.batchSize,
                                      *project[z]->getOutputPtr(0),
                                      gParams.projectedSize * sizeof(value_type) * gParams.batchSize,
                                      cudaMemcpyDeviceToDevice, layerStream[lastL]));
            }
            cudaStreamSynchronize(layerStream[lastL]);
        }
        // return cell to initial state for another run
        // move ptr to the beginning of the time sequence data
        *lstm[0]->getInputPtr(0) = startInputPtr;

        // return cell ptrs to original state:
        for (uint32_t z = 0; z < gParams.layers; z++)
        {
            if (gParams.seqSize % 2 == 0)
            {
                std::swap(*lstm[z]->getOutputPtr(1), *lstm[z]->getInputPtr(iCellId));
            }
            else
            {
                std::swap(*project[z]->getInputPtr(0), *lstm[z]->getOutputPtr(0));
            }
        }
    }
    cudaProfilerStop();
    for (int i = 0; i < 3; i++)
        avgTime[i] /= gParams.avgRuns;

    // swap some pointers back
    /*    std::swap(*lstmp[lastL]->getOutputPtr(0), *logits.getInputPtr(0));*/
    // DMA the output from the GPU
    logits.copyDataFromDevice({0, 1}, stream);
    // in case of debug
    lstm[lastL]->copyDataFromDevice({0, 1, 2, 3}, stream);
    project[lastL]->copyDataFromDevice({0, 1}, stream);

    for (uint32_t z = 0; z < gParams.layers; z++)
    {
        CHECK(cudaStreamDestroy(layerStream[z]));
        if (z < lastL)
        {
            CHECK(cudaStreamDestroy(copyStream[z]));
        }
    }

    for (uint32_t t = 0; t < gParams.seqSize; t++)
        for (uint32_t z = 0; z < gParams.layers; z++)
        {
            CHECK(cudaEventDestroy(layerEvent[t][z]));
            CHECK(cudaEventDestroy(copyEvent[t][z]));
        }

    cudaStreamSynchronize(stream);
    CHECK(cudaEventDestroy(start1));
    CHECK(cudaEventDestroy(start2));
    CHECK(cudaEventDestroy(end));
}

float getMedian(std::vector<float>& runTime)
{
    std::sort(runTime.begin(), runTime.end());
    float total_median;
    if (gParams.iterations % 2 == 1)
    {
        total_median = runTime[gParams.iterations / 2];
    }
    else
    {
        total_median = (runTime[gParams.iterations / 2] + runTime[gParams.iterations / 2 + 1]) * 0.5F;
    }
    return total_median;
}

template <class value_type>
void lstmpBytesFlops(std::map<std::string, Weights>& weightMap, size_t& bytes, size_t& ops)
{
    size_t rnnweightCount = gParams.layers * (gParams.dataSize + gParams.projectedSize) * 4 * gParams.cellSize;
    size_t projweightCount = gParams.layers * gParams.projectedSize * gParams.cellSize;
    size_t biasweightCount = gParams.layers * 4 * gParams.cellSize;

    if (rnnweightCount != (size_t) weightMap["rnnweight"].count)
    {
        gLogError << rnnweightCount << " " << static_cast<size_t>(weightMap["rnnweight"].count) << std::endl;
        FatalError("rnnweightCount: Expected sizes missmatch");
    }

    if (projweightCount != (size_t) weightMap["rnnweightp"].count)
    {
        gLogError << projweightCount << " " << static_cast<size_t>(weightMap["rnnweightp"].count) << std::endl;
        FatalError("projweightCount: Expected sizes missmatch");
    }

    if (biasweightCount != (size_t) weightMap["rnnbias"].count)
    {
        gLogError << biasweightCount << " " << static_cast<size_t>(weightMap["rnnbias"].count) << std::endl;
        FatalError("biasweightCount: Expected sizes missmatch");
    }

    // inputs per time step (data between LSTM and projection layer is counted)
    size_t inputsOutputsCount = gParams.batchSize * gParams.layers * (gParams.cellSize + gParams.dataSize + gParams.projectedSize + gParams.cellSize + gParams.cellSize + gParams.projectedSize);

    bytes = (rnnweightCount + projweightCount + biasweightCount + inputsOutputsCount) * sizeof(value_type);
    ops = 2ull * gParams.batchSize * gParams.layers * ((gParams.dataSize + gParams.projectedSize) * 4 + gParams.projectedSize) * gParams.cellSize; // 4 - number of parameters of LSTM
}

template <class value_type>
void logitsBytesFlops(std::map<std::string, Weights>& weightMap, size_t& bytes, size_t& ops)
{
    size_t logitsweightCount = gParams.projectedSize * OUTPUT_SIZE;
    size_t logitsbiasCount = OUTPUT_SIZE;

    if (logitsweightCount != (size_t) weightMap["rnnfcw"].count)
    {
        gLogError << logitsweightCount << " " << static_cast<size_t>(weightMap["rnnfcw"].count) << std::endl;
        FatalError("logitsweightCount: Expected sizes missmatch");
    }

    if (logitsbiasCount != (size_t) weightMap["rnnfcb"].count)
    {
        gLogError << logitsbiasCount << " " << static_cast<size_t>(weightMap["rnnfcb"].count) << std::endl;
        FatalError("logitsbiasCount: Expected sizes missmatch");
    }

    // inputs per time step (data between LSTM and projection layer is not counted)
    size_t inputsOutputsCount;
    if (!gParams.sm)
    {
        inputsOutputsCount = gParams.batchSize * (gParams.projectedSize + OUTPUT_SIZE);
    }
    else
    {
        inputsOutputsCount = gParams.batchSize * (gParams.projectedSize + 3 * OUTPUT_SIZE);
    }

    bytes = (logitsweightCount + logitsbiasCount + inputsOutputsCount) * sizeof(value_type);
    ops = 0ull; // assume bandwidth limited;
}

template <class value_type>
bool cmp(const value_type& a, const value_type& b)
{
    Convert<float> toFloat;
    return toFloat(a) < toFloat(b);
}

template <DataType nvInferType>
bool doInference(ICudaEngine** engineLSTM, ICudaEngine** enginePROJECT, ICudaEngine* engineLOGITS, Dataset& input,
                 std::vector<std::string>& expected, std::map<std::string, Weights>& weightMap,
                 const CUDADeviceTroughput& deviceThroughput)
{
    double perplexity = 0.0;
    typedef typename InferTypeToValue<nvInferType>::value_type value_type;
    std::vector<IExecutionContext*> contextLSTM(gParams.layers);
    std::vector<IExecutionContext*> contextPROJECT(gParams.layers);
    for (uint32_t z = 0; z < gParams.layers; z++)
    {
        contextLSTM[z] = engineLSTM[z]->createExecutionContext();
        contextPROJECT[z] = enginePROJECT[z]->createExecutionContext();
    }
    IExecutionContext* contextLOGITS = engineLOGITS->createExecutionContext();
    std::vector<BindData<value_type>*> lstm(gParams.layers);
    std::vector<BindData<value_type>*> project(gParams.layers);
    // initialize inputs only once
    for (uint32_t z = 0; z < gParams.layers; z++)
    {
        uint32_t inputSize = 1 * gParams.batchSize * (gParams.dataSize + gParams.projectedSize);
        if (z == 0)
            inputSize *= gParams.seqSize;
        lstm[z] = new BindData<value_type>(engineLSTM[z],
                                           {INPUT_BLOB_NAME,
#ifdef INIT_HIDDEN_IN
                                            HIDDEN_IN_BLOB_NAME,
#endif
                                            CELL_IN_BLOB_NAME,
                                            LSTM_OUT_BLOB_NAME,
                                            CELL_OUT_BLOB_NAME},
                                           {inputSize,
#ifdef INIT_HIDDEN_IN
                                            1 * gParams.batchSize * gParams.cellSize,
#endif
                                            1 * gParams.batchSize * gParams.cellSize,
                                            1 * gParams.batchSize * gParams.cellSize,
                                            1 * gParams.batchSize * gParams.cellSize},
                                           2, // numOutputs
                                           z);
        project[z] = new BindData<value_type>(enginePROJECT[z],
                                              {PROJECT_IN_BLOB_NAME,
                                               PROJECT_OUT_BLOB_NAME},
                                              {1 * gParams.batchSize * gParams.cellSize,
                                               gParams.batchSize * gParams.projectedSize},
                                              1, // numOutputs
                                              z);
    }
    uint32_t logitsBatchSize = (gParams.perplexity || IsDebug) ? gParams.batchSize * gParams.seqSize : gParams.batchSize;
    BindData<value_type> logits(engineLOGITS,
                                {LOGITS_IN_BLOB_NAME,
                                 OUTPUT_BLOB_NAME},
                                {1 * logitsBatchSize * gParams.projectedSize,
                                 logitsBatchSize * OUTPUT_SIZE},
                                1 // numOutputs
    );
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    auto embed = weightMap["embed"];
    std::vector<uint32_t> x(gParams.seqSize * gParams.batchSize, 0);
    std::vector<uint32_t> y(gParams.seqSize * gParams.batchSize, 0);

    // Input / output strings:
    std::vector<std::vector<std::string>> inputStrings(gParams.batchSize, std::vector<std::string>(gParams.seqSize));
    std::vector<std::vector<std::string>> outputStrings(gParams.batchSize, std::vector<std::string>(gParams.iterations));

    std::string outputTitle = (!gParams.sm) ? "Output FC per time step" : "Output FC + Softmax per time step";
    std::vector<std::vector<float>> runTime(3, std::vector<float>(gParams.iterations, 0));
    std::vector<double> loss(gParams.iterations);
    std::fstream fs;
    if (IsDebug)
        fs.open("logits.txt", std::fstream::out);

    // Send init data to device
    for (uint32_t z = 0; z < gParams.layers; z++)
    {
#ifdef INIT_HIDDEN_IN
        // 3 inputs
        lstm[z]->copyDataToDevice({0, 1, 2}, stream);
#else
        lstm[z]->copyDataToDevice({0, 1}, stream);
#endif
        project[z]->copyDataToDevice({0}, stream);
    }
    logits.copyDataToDevice({0}, stream);

    // allocate a buffer to handle the state between iterations
    // (need this since each iteration is launched avgRuns times for more precise timings)
    value_type* stateBuffer = nullptr;
    size_t stateBufferSize = (gParams.cellSize + gParams.projectedSize) * gParams.layers * gParams.batchSize * sizeof(value_type);
    cudaMalloc(&stateBuffer, stateBufferSize);
    cudaMemset(stateBuffer, 0, stateBufferSize);

    for (uint32_t it = 0; it < gParams.iterations; it++)
    {
        // populate x with first batch of hot vectors
        input.iterateOnce(x, y);
        gLogInfo << "Iteration " << it << std::endl;
        for (uint32_t i = 0; i < gParams.seqSize; i++)
        {
            for (uint32_t n = 0; n < gParams.batchSize; n++)
            {
                std::stringstream ss;
                uint32_t tokenId = x[n * gParams.seqSize + i];
                ss << input.getVocabularyPtr()->getToken(tokenId) << "(" << tokenId << ")";
                inputStrings[n][i] = ss.str();
                std::copy(reinterpret_cast<const value_type*>(embed.values) + tokenId * gParams.dataSize,
                          reinterpret_cast<const value_type*>(embed.values) + tokenId * gParams.dataSize + gParams.dataSize,
                          reinterpret_cast<value_type*>(lstm[0]->data()[0]) + n * (gParams.dataSize + gParams.projectedSize)
                              + i * gParams.batchSize * (gParams.dataSize + gParams.projectedSize));
            }
        }
        // send embed data to device
        lstm[0]->copyDataToDevice({0}, stream);
        CHECK(cudaStreamSynchronize(stream));
        float avgTime[3] = {0.f};
        stepOnce<value_type>(lstm.data(), project.data(), logits, stream, contextLSTM.data(), contextPROJECT.data(), contextLOGITS,
                             stateBuffer, avgTime);
        //cudaMemset(stateBuffer, 0, stateBufferSize);
        for (int i = 0; i < 3; i++)
            runTime[i][it] = avgTime[i];

        gLogInfo << "LSTMP time = " << avgTime[0] << " ms\n";
        gLogInfo << outputTitle << " = " << avgTime[1] << " ms\n";

        value_type* probabilities = reinterpret_cast<value_type*>(logits.data()[1]);

        double avgLoss = 0.0;

        Convert<float> toFloat;
        double halfEpsilon = (nvInferType == DataType::kHALF) ? 1.e-7 : 1.e-37;

        size_t outputStride = (gParams.perplexity || IsDebug) ? gParams.seqSize - 1 : 0;
        outputStride *= OUTPUT_SIZE * gParams.batchSize;

        for (uint32_t n = 0; n < gParams.batchSize; n++)
        {
            ptrdiff_t idx = std::max_element(probabilities + outputStride + n * OUTPUT_SIZE, probabilities + outputStride + n * OUTPUT_SIZE + OUTPUT_SIZE, cmp<value_type>) - probabilities - outputStride - n * OUTPUT_SIZE;
            std::stringstream ss;
            ss << input.getVocabularyPtr()->getToken(idx) << "(" << idx << ")";
            outputStrings[n][it] = ss.str();
            if (gParams.perplexity)
            {
                std::stringstream ss;
                for (uint32_t t = 0; t < gParams.seqSize; t++)
                {
                    uint32_t tokenId = y[n * gParams.seqSize + t];
                    double val = std::max((double) toFloat(probabilities[OUTPUT_SIZE * gParams.batchSize * t + n * OUTPUT_SIZE + tokenId]), halfEpsilon);
                    avgLoss += std::log(val) / gParams.seqSize;
                    ss << input.getVocabularyPtr()->getToken(tokenId) << "(" << tokenId << ")"
                       << "  " << std::exp(-std::log(val)) << " ";
                }
            }
            else
            {
                // Ground truth ID
                uint32_t tokenId = y[(n + 1) * gParams.seqSize - 1];
                if (gParams.sm)
                {
                    double val = std::max((double) toFloat(probabilities[outputStride + n * OUTPUT_SIZE + tokenId]), halfEpsilon);
                    avgLoss += std::log(val);
                }
            }
        }

        if (IsDebug)
        {
            uint32_t seqSize = (gParams.perplexity || IsDebug) ? gParams.seqSize : 1;
            // Debug dump logits
            //uint32_t debugStride = gParams.projectedSize;
            uint32_t debugStride = OUTPUT_SIZE;
            fs << "Iter " << it << std::endl;
            for (uint32_t t = 0; t < seqSize; t++)
            {
                fs << "Time " << t << std::endl;
                for (uint32_t n = 0; n < gParams.batchSize; n++)
                {
                    fs << "Batch " << n << std::endl;
                    for (uint32_t tokenId = 0; tokenId < std::min(debugStride, uint32_t(1024)); tokenId++)
                    {
                        fs << probabilities[debugStride * gParams.batchSize * t + n * debugStride + tokenId] << "\n";
                    }
                    fs << std::endl;
                }
                fs << std::endl;
            }
        }
        loss[it] = avgLoss / gParams.batchSize;

        if (gParams.sm)
        {
            //gLogInfo << "Perplexity at iteration " << it << " = " << std::exp( -avgLoss / gParams.batchSize) << std::endl;
            perplexity += avgLoss / gParams.batchSize;
        }

        //print word streams
        for (uint32_t n = 0; n < gParams.batchSize; n++)
        {
            gLogInfo << "Batch " << n << " input: ";
            for (auto i : inputStrings[n])
                gLogInfo << i << " ";
            gLogInfo << "\nBatch " << n << " output: " << outputStrings[n][it] << std::endl;
        }
    }

    gLogInfo << "Median LSTMP runtime is " << getMedian(runTime[0]) << " ms" << std::endl;
    gLogInfo << "Median " << outputTitle << " runtime is " << getMedian(runTime[1]) << " ms" << std::endl;
    float medianTotalTime = getMedian(runTime[2]);
    gLogInfo << "Median total runtime is " << medianTotalTime << " ms achiving " << int(gParams.seqSize / medianTotalTime * 1000.0f * gParams.batchSize) << " WPS (Words Per Second)" << std::endl;

    // TODO: Fix for half
    size_t bytesLSTMP, flopsLSTMP, bytesLOGITS, flopsLOGITS;
    lstmpBytesFlops<value_type>(weightMap, bytesLSTMP, flopsLSTMP);
    logitsBytesFlops<value_type>(weightMap, bytesLOGITS, flopsLOGITS);
    double gbFactor = 1.0 / getMedian(runTime[0]) / 1000000.0 * gParams.seqSize;
    std::string gOPS = (std::is_same<half1, value_type>::value) ? "GHFLOPS" : "GFLOPS";
    double halfFactor = (std::is_same<half1, value_type>::value) ? 2.0 : 1.0;
    gLogInfo << "LSTMP per time step : " << flopsLSTMP * gbFactor << "(" << deviceThroughput.fp32flops * halfFactor / 1000000000
             << ") " << gOPS << " at " << bytesLSTMP * gbFactor << "("
             << deviceThroughput.achievableMemoryBandwidth / 1000000000 << ") GB/s" << std::endl;
    gbFactor = 1.0 / getMedian(runTime[1]) / 1000000.0;
    gLogInfo << outputTitle << " : " << bytesLOGITS * gbFactor << "("
             << deviceThroughput.achievableMemoryBandwidth / 1000000000 << ") GB/s" << std::endl;

    if ((gParams.sm || gParams.perplexity) && !IsDebug)
    {
        if (!gParams.perplexity)
        {
            gLogWarning << "Perplexity calculation is based on final timestep only" << std::endl;
            gLogWarning << "Use '--perplexity' option for precise results (affects performance)" << std::endl;
        }

        gLogInfo << "log_perplexity (perplexity) at each iteration:\n";
        for (uint32_t it = 0; it < loss.size(); it++)
            gLogInfo << it << ": " << std::setprecision(4) << -loss[it] << " (" << std::exp(-loss[it]) << ") ... ";
        gLogInfo << std::endl;

        std::vector<double> lossAcum(loss.size());
        std::partial_sum(loss.begin(), loss.end(), lossAcum.begin(), std::plus<double>());

        perplexity = std::exp(-lossAcum.back() / gParams.iterations);
        double stdDev = 0.0;
        for (auto v : loss)
            stdDev += (std::exp(-v) - perplexity) * (std::exp(-v) - perplexity);
        stdDev = std::sqrt(stdDev / (loss.size() - 1));
        gLogInfo << "Perplexity (stddev) from average loss = " << perplexity
                 << " (" << stdDev << ")"
                 << std::endl;

        gLogInfo << "Moving average log_perplexity (perplexity) at each iteration:\n";
        for (uint32_t it = 0; it < loss.size(); it++)
            gLogInfo << it << ": " << std::setprecision(4) << -lossAcum[it] / (it + 1) << " (" << std::exp(-lossAcum[it] / (it + 1)) << ") ... ";
        gLogInfo << std::endl;
    }
    else
    {
        gLogWarning << "Should enable SoftMax or provide --perplexity flag for perplexity calculation" << std::endl;
    }

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    for (uint32_t z = 0; z < gParams.layers; z++)
    {
        delete lstm[z];
        contextLSTM[z]->destroy();
        delete project[z];
        contextPROJECT[z]->destroy();
    }
    contextLOGITS->destroy();
    cudaFree(stateBuffer);

    // data verification is only implemented for batch 1
    if (gParams.batchSize == 1)
    {
        for (uint32_t it = 0; it < std::min(expected.size(), (size_t) gParams.iterations); it++)
        {
            if (outputStrings[0][it] != expected[it])
            {
                gLogError << it << " " << outputStrings[0][it] << " != " << expected[it] << std::endl;
                return false;
            }
        }
        gLogInfo << "Verification is successful" << std::endl;
    }
    else
    {
        gLogInfo << "Batch > 1: skipping verification step" << std::endl;
    }
    return true;
}

int main(int argc, char** argv)
{
    if (!parseArgs(argc, argv, gParams))
    {
        printUsage(gParams);
        return EXIT_FAILURE;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    cudaSetDevice(gParams.device);

    cudaDeviceProp deviceProp;
    getDeviceInfo(gParams.device, &deviceProp);
    gLogInfo << "Running on CUDA device: " << deviceProp.name
             << " (" << deviceProp.clockRate / 1000000.0F << " GHz, "
             << deviceProp.multiProcessorCount << " SMs"
             << ", mem " << deviceProp.memoryClockRate / 1000000.0F << " GHz, "
             << ((deviceProp.ECCEnabled != 0) ? "ECC enabled" : "ECC disabled")
             << ", "
             << deviceProp.memoryBusWidth << " bits"
             << ", Compute Capability " << deviceProp.major << "." << deviceProp.minor << ")" << std::endl;

    CUDADeviceTroughput deviceThroughput(gParams.device, gParams.overrideGPUClocks, gParams.overrideMemClocks,
                                         gParams.overrideAchievableMemoryBandwidthRatio);
    gLogInfo << "CUDA device throughput: " << deviceThroughput << std::endl;

    Vocabulary vocabulary;
    vocabulary.fromFile(locateFile2(gParams.dataDir, gParams.vocabFile));
    testVocabulary(vocabulary);
    Dataset inputDataset(gParams.batchSize, gParams.seqSize, vocabulary);
    inputDataset.parseFile(locateFile2(gParams.dataDir, gParams.inputsFile));

    std::map<std::string, Weights> weightMap = (gParams.half2) ? loadWeights<DataType::kHALF>(gParams.dataDir, gParams.weightsFile) : loadWeights<DataType::kFLOAT>(gParams.dataDir, gParams.weightsFile);
    gLogInfo << "Preprocessing weights" << std::endl;
    // must preprocess before engine creation
    if (gParams.half2)
    {
        preprocessWeights<DataType::kHALF>(weightMap, gParams.dataSize, gParams.projectedSize, gParams.cellSize,
                                           gParams.layers);
    }
    else
    {
        preprocessWeights<DataType::kFLOAT>(weightMap, gParams.dataSize, gParams.projectedSize, gParams.cellSize,
                                            gParams.layers);
    }

    gLogInfo << "Building the engine" << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
    std::vector<ICudaEngine*> engineLSTM(gParams.layers);
    std::vector<ICudaEngine*> enginePROJECT(gParams.layers);
    ICudaEngine* engineLOGITS;

    bool pass{false};

    if (gParams.half2)
    {
        for (uint32_t z = 0; z < gParams.layers; z++)
        {
            engineLSTM[z] = APIToModelLSTM<DataType::kHALF>(weightMap, z);
            enginePROJECT[z] = APIToModelPROJECT<DataType::kHALF>(weightMap, z);
        }
        engineLOGITS = APIToModelLOGITS<DataType::kHALF>(weightMap);
        pass = doInference<DataType::kHALF>(engineLSTM.data(), enginePROJECT.data(), engineLOGITS, inputDataset,
                                            gParams.expected.at(gParams.weightsFile), weightMap, deviceThroughput);
    }
    else
    {
        for (uint32_t z = 0; z < gParams.layers; z++)
        {
            engineLSTM[z] = APIToModelLSTM<DataType::kFLOAT>(weightMap, z);
            enginePROJECT[z] = APIToModelPROJECT<DataType::kFLOAT>(weightMap, z);
        }
        engineLOGITS = APIToModelLOGITS<DataType::kFLOAT>(weightMap);
        pass = doInference<DataType::kFLOAT>(engineLSTM.data(), enginePROJECT.data(), engineLOGITS, inputDataset,
                                             gParams.expected.at(gParams.weightsFile), weightMap, deviceThroughput);
    }

    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }
    // destroy the engine
    for (uint32_t z = 0; z < gParams.layers; z++)
    {
        if (engineLSTM[z])
            engineLSTM[z]->destroy();
        if (enginePROJECT[z])
            enginePROJECT[z]->destroy();
    }
    if (engineLOGITS)
        engineLOGITS->destroy();
    if (runtime)
        runtime->destroy();

    return gLogger.reportTest(sampleTest, pass);
}
