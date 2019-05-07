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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "NvUffParser.h"
#include "NvUtils.h"
#include "common.h"
#include "logger.h"

using namespace nvuffparser;
using namespace nvinfer1;

static int gUseDLACore{-1};

const std::string gSampleName = "TensorRT.sample_uff";

// The _GB literal operator is defined in common/common.h
constexpr size_t MAX_WORKSPACE = 1_GB;

// Convert from a vector of strings to a vector of either floats or ints.
template <typename T>
std::vector<T> convertTo(const std::vector<std::string>& stringVector)
{
    std::vector<T> convertedVector(stringVector.size());
    std::transform(stringVector.begin(), stringVector.end(), convertedVector.begin(),
                   [](const std::string& elem) {
                       return (std::is_same<T, int>::value) ? std::stoi(elem) : std::stof(elem);
                   });
    return convertedVector;
}

// Compute memory requirements of input/ouptut bindings.
std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = samplesCommon::volume(dims) * batchSize;
        sizes.emplace_back(eltCount, dtype);
    }
    return sizes;
}

// Create a buffer on the GPU to hold engine inputs.
void* createInputCudaBuffer(int64_t eltCount, DataType dtype, const std::string& inputFileName)
{
    if (samplesCommon::getElementSize(dtype) != sizeof(float))
    {
        gLogger.log(ILogger::Severity::kERROR, "In createInputCudaBuffer: Expected 32-bit type.");
        abort();
    }

    std::ifstream inputFileStream{inputFileName};
    if (!inputFileStream.is_open())
        throw std::runtime_error("Could not read input file: " + inputFileName + ". Check input path.");

    std::string rawInput{std::istreambuf_iterator<char>{inputFileStream}, std::istreambuf_iterator<char>{}};
    std::vector<float> float_vect{convertTo<float>(samplesCommon::splitString(rawInput))};

    if (eltCount > static_cast<int64_t>(float_vect.size()))
    {
        std::string errorMessage{"Requested network input size (" + std::to_string(eltCount)
                                 + ") is larger than number of values (" + std::to_string(float_vect.size())
                                 + ") in input file. Please provide valid input dimensions or batch size."};
        gLogger.log(ILogger::Severity::kERROR, errorMessage.c_str());
        abort();
    }

    size_t memSize = eltCount * samplesCommon::getElementSize(dtype);
    void* deviceMem = samplesCommon::safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, float_vect.data(), memSize, cudaMemcpyHostToDevice));
    return deviceMem;
}

// Write one engine output into a single file.
void writeOutput(int64_t eltCount, DataType dtype, void* buffer, std::string outputFileName)
{
    if (samplesCommon::getElementSize(dtype) != sizeof(float))
    {
        gLogger.log(ILogger::Severity::kERROR, "In createInputCudaBuffer: Expected 32-bit type.");
        abort();
    }

    size_t memSize = eltCount * samplesCommon::getElementSize(dtype);
    std::vector<float> outputs;
    try
    {
        outputs.resize(eltCount);
    }
    catch (const std::bad_alloc&)
    {
        std::string errorMessage = "Ran out of host memory when trying to allocate host output buffer of length "
            + std::to_string(eltCount) + ". Try decreasing batch size.";
        gLogger.log(ILogger::Severity::kERROR, errorMessage.c_str());
        abort();
    }
    CHECK(cudaMemcpy(outputs.data(), buffer, memSize, cudaMemcpyDeviceToHost));

    std::ofstream outfile{outputFileName};
    if (!outfile.is_open())
        throw std::runtime_error("Could not write to output file: " + outputFileName + ". Is it writable?");

    // Write outputs
    std::for_each(outputs.begin(), outputs.end() - 1,
                  [&outfile](float elem) {
                      outfile << elem << ",";
                  });
    // The last element should not be followed by a comma.
    outfile << outputs.back();
    outfile.close();
}

ICudaEngine* loadModelAndCreateEngine(const std::string& trtFile, int maxBatchSize,
                                      IUffParser* parser, int precision)
{
    IBuilder* builder = createInferBuilder(gLogger);
    assert(builder != nullptr);
    INetworkConfig* config = builder->createNetworkConfig();
    INetworkDefinition* network = builder->createNetwork();
    // Check that precision is either 0 or 1 i.e. a 1-bit quantity.
    if ((precision & 1) != precision)
    {
        gLogWarning << "Precision " + std::to_string(precision) + " unrecognized. Defaulting to FP32." << std::endl;
        precision = 0;
    }
    if (!parser->parse(trtFile.c_str(), *network, static_cast<DataType>(precision)))
    {
        gLogError << "Failure while parsing UFF file" << std::endl;
        return nullptr;
    }

    // Create the engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(MAX_WORKSPACE);
    if (precision == 1)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    samplesCommon::enableDLA(builder, config, gUseDLACore);

    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        gLogError << "Unable to create engine" << std::endl;
        return nullptr;
    }

    // Clean up the network and parser
    network->destroy();
    builder->destroy();
    config->destroy();
    return engine;
}

void execute(ICudaEngine& engine, std::vector<std::string> inputFileNames, std::vector<std::string> outputFileNames, int batchSize)
{
    IExecutionContext* context = engine.createExecutionContext();

    unsigned int nbBindings = engine.getNbBindings();
    if (nbBindings != inputFileNames.size() + outputFileNames.size())
    {
        throw std::runtime_error("Engine has " + std::to_string(nbBindings) + " bindings, but received " + std::to_string(inputFileNames.size()) + " input files and " + std::to_string(outputFileNames.size()) + " output files.");
    }

    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);
    // Create inputs
    std::vector<int> bindingIdxInputs;
    for (unsigned int i = 0; i < nbBindings; ++i)
    {
        auto bufferSize = buffersSizes[i];
        if (engine.bindingIsInput(i))
        {
            bindingIdxInputs.push_back(i);
            buffers[i] = createInputCudaBuffer(bufferSize.first, bufferSize.second, inputFileNames[i]);
        }
        else
        {
            buffers[i] = samplesCommon::safeCudaMalloc(bufferSize.first * samplesCommon::getElementSize(bufferSize.second));
        }
    }
    // Timing Code.
    int outcount = 0;
    float total = 0.f;
    auto t_start = std::chrono::high_resolution_clock::now();
    // Execute the engine.
    context->execute(batchSize, &buffers[0]);
    auto t_end = std::chrono::high_resolution_clock::now();
    total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    gLogInfo << "Average over 1 run is: " << total << " ms." << std::endl;

    // Write outputs to disk.
    for (unsigned int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
    {
        if (engine.bindingIsInput(bindingIdx))
            continue;
        auto bufferSizesOutput = buffersSizes[bindingIdx];
        writeOutput(bufferSizesOutput.first, bufferSizesOutput.second, buffers[bindingIdx], outputFileNames[outcount]);
        outcount++;
    }
    // Clean up.
    for (auto const& value : bindingIdxInputs)
        CHECK(cudaFree(buffers[value]));

    for (unsigned int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();
}

void displayHelp()
{
    std::cout
        << "Usage: ./sample_uff UFF_FILE PRECISION OUTPUT_LAYER(S) INPUT_LAYER(S) INPUT_DIRECTORY INPUT_DIMENSION(S) OUTPUT_DIRECTORY [--useDLACore=N]\n"
        << "\n\tUFF_FILE: Path to a UFF file.\n"
        << "\tPRECISION: Precision with which to run the engine. 0 for FP32, 1 for FP16.\n"
        << "\tOUTPUT_LAYER(S): Name(s) of output layer(s). Comma-separated (with NO spaces) for multiple outputs.\n"
        << "\tINPUT_LAYER(S): Name(s) of input layer(s). Comma-separated (with NO spaces) for multiple inputs.\n"
        << "\tINPUT_DIRECTORY: Directory containing inputs in text files using the naming scheme inp_#.txt where # is the input number.\n"
        << "\tINPUT_DIMENSION(S): Dimensions of input(s). Dimensions for a single input are comma-separated.\n"
        << "\t\tMultiple inputs can be specified by space separating dimensions.\n"
        << "\t\tFor example, for 2 inputs: 1,3,28,28 1,3,56,56\n"
        << "\tOUTPUT_DIRECTORY: Directory where the output files should be written to in text files using the naming scheme outtrt_#.txt where # is the input number.\n"
        << "\t--useDLACore=N: Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform.\n"
        << "\n\tExample: ./sample_uff vgg16/vgg16.uff 0 pool5,conv5_3/BiasAdd,prob input vgg16/inputs/ 4,3,224,224 vgg16/outputs/ --useDLACore=0\n"
        << std::endl;
}

int main(int argc, char** argv)
{
    // Usage: ./sample_uff ufffile precision outnames innames inputvals inpdims1 [inpdims2 inpdims3 ...]   # inpdims1 are comma-separated ints
    // outnames and innames are comma separated (for multiple outputs and inputs)
    // Sample usage: ./sample_uff mydumpvgg16/a.uff 0 pool5,conv5_3/BiasAdd,prob input mydumpvgg16/0/ 4,3,224,224 vgg16/outputs/

    if (argc < 8)
    {
        displayHelp();
        return EXIT_FAILURE;
    }

    gUseDLACore = samplesCommon::parseDLA(argc, argv);

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    std::string fileName = argv[1];

    int maxBatchSize;
    auto parser = createUffParser();
    // Register input/output
    auto outputs = samplesCommon::splitString(argv[3]);
    for (const auto& output : outputs)
        parser->registerOutput(output.c_str());
    auto inputs = samplesCommon::splitString(argv[4]);
    unsigned int batchSize = 0;
    for (unsigned i = 0; i < inputs.size(); i++)
    {
        std::vector<int> input{convertTo<int>(samplesCommon::splitString(argv[6 + i]))};
        Dims inputDims;
        inputDims.nbDims = input.size() - 1;
        for (unsigned dim = input.size() - 1; dim > 0; --dim)
        {
            inputDims.d[dim - 1] = input[dim];
            inputDims.type[dim - 1] = DimensionType::kSPATIAL;
        }
        inputDims.type[0] = DimensionType::kCHANNEL;
        parser->registerInput((inputs[i]).c_str(), inputDims, UffInputOrder::kNCHW);
        if (batchSize && static_cast<unsigned int>(input[0]) != batchSize)
        {
            gLogError << "All inputs should have the same batch size." << std::endl;
            return gLogger.reportFail(sampleTest);
        }
        batchSize = input[0];
    }

    maxBatchSize = batchSize;

    // Get input/output filenames based on input/output names.
    std::vector<std::string> outputFileNames;
    {
        int index = 0;
        for (auto& input : inputs)
            input = argv[5] + std::string("inp_") + std::to_string(index++) + std::string(".txt");

        for (unsigned index = 0; index < outputs.size(); index++)
        {
            outputFileNames.push_back(argv[7] + std::string("outtrt_") + std::to_string(index) + std::string(".txt"));
        }
    }

    ICudaEngine* engine = loadModelAndCreateEngine(fileName, maxBatchSize, parser, std::stoi(argv[2]));

    if (!engine)
    {
        gLogError << "Model load failed" << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    unsigned int nbBindings = engine->getNbBindings();
    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(*engine, nbBindings, 1);

    parser->destroy();

    execute(*engine, inputs, outputFileNames, batchSize);
    engine->destroy();

    shutdownProtobufLibrary();

    return gLogger.reportPass(sampleTest);
}
