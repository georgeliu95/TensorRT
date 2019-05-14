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

//!
//! sampleMovieLens.cpp
//! This file contains the implementation of the MovieLens sample. It creates the network using
//! the MLP NCF Uff model.
//! It can be run with the following command line:
//! Command: ./sample_movielens [-h or --help] [-b NUM_USERS] [--useDLACore=<int>] [--verbose]
//!

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

const std::string gSampleName = "TensorRT.sample_movielens";

// The OutputParams struct holds intermediate/final outputs generated by the MovieLens structure per user.
struct OutputParams
{
    int32_t userId;                                         // The user Id per batch.
    int32_t expectedPredictedMaxRatingItem;                 // The Expected Max Rating Item per user (inference ground truth).
    float expectedPredictedMaxRatingItemProb;               // The Expected Max Rating Probability. (inference ground truth).
    std::vector<int32_t> allItems;                          // All inferred items per user.
    std::vector<std::pair<int32_t, float>> itemProbPairVec; // Expected topK items and prob per user.
};                                                          // struct pargs

//!
//! \brief The SampleMovieLensParams structure groups the additional parameters required by
//!         the MovieLens sample.
//!
struct SampleMovieLensParams : public samplesCommon::UffSampleParams
{
    int32_t embeddingVecSize;
    int32_t numUsers;            // Total number of users. Should be equal to ratings file users count.
    int32_t topKMovies;          // TopK movies per user.
    int32_t numMoviesPerUser;    // The number of movies per user.
    std::string ratingInputFile; // The input rating file.
    bool strict;                 // Option to run with strict type requirements.

    // The below structures are used to compare the predicted values to inference (ground truth)
    std::map<int32_t, std::vector<int32_t>> userToItemsMap;                              // Lookup for inferred items for each user.
    std::map<int32_t, std::vector<std::pair<int32_t, float>>> userToExpectedItemProbMap; // Lookup for topK items and probs for each user.
    std::vector<OutputParams> outParamsVec;
};

//!
//! \brief  The SampleMovieLens class implements the MovieLens sample
//!
//! \details It creates the network using a uff model
//!
class SampleMovieLens
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleMovieLens(const SampleMovieLensParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief Builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    //!
    //! \brief Used to clean up any state created in the sample class
    //!
    bool teardown();

private:
    //!
    //! \brief Parses a Uff model for a MLP NCF model, creates a TensorRT network, and builds a TensorRT engine.
    //!
    void constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                          SampleUniquePtr<nvinfer1::INetworkConfig>& config,
                          SampleUniquePtr<nvuffparser::IUffParser>& parser);
    //!
    //! \brief Copies a batch of input data from SampleMovieLensParams into managed input buffers
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Helper function to read the next line of the MovieLens dataset
    //!        .csv file and return the contents of the line after the delimeter.
    std::string readNextLine(std::ifstream& file, char delim);

    //!
    //! \brief Extracts needed dataset values for a single user in the MovieLens,
    //!        dataset .csv file, and populates the corresponding ground truth data struct
    //!
    void readInputSample(std::ifstream& file, OutputParams& outParams, std::string line);

    //!
    //! \brief Parses the MovieLens dataset and populates the SampleMovieLensParams data structure
    //!
    void parseMovieLensData();

    //!
    //! \brief Prints the expected recommendation results (ground truth)
    //!        from the MovieLens dataset for a given user
    //!
    void printOutputParams(OutputParams& outParams);

    //!
    //! \brief Verifies the inference output with ground truth and logs the results
    //!
    bool verifyOutput(uint32_t* userInputPtr, uint32_t* /*itemInputPtr*/, uint32_t* topKItemNumberPtr, float* topKItemProbPtr);

    SampleMovieLensParams mParams;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr}; //!< The TensorRT engine used to run the network
};

//!
//! \brief Creates the network, configures the builder and creates
//! the network engine
//!
//! \details This function creates the MLP NCF network by parsing the Uff model
//! and builds the engine that will be used to generate recommendations (mEngine)
//!
//! \return Returns true if the engine was created successfully and false
//! otherwise
//!
bool SampleMovieLens::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }
    auto config = SampleUniquePtr<nvinfer1::INetworkConfig>(builder->createNetworkConfig());
    if (!config)
    {
        return false;
    }
    auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!parser)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1_GB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    config->setFlag(BuilderFlag::kSTRICT_TYPES);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    constructNetwork(builder, network, config, parser);

    if (!mEngine)
    {
        return false;
    }

    return true;
}

//!
//! \brief Parses a Uff model for a MLP NCF model, creates a TensorRT network, and builds a TensorRT engine.
//!
void SampleMovieLens::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                       SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                                       SampleUniquePtr<nvinfer1::INetworkConfig>& config,
                                       SampleUniquePtr<nvuffparser::IUffParser>& parser)
{

    nvinfer1::Dims inputIndices;
    inputIndices.nbDims = 3;
    inputIndices.d[0] = mParams.numMoviesPerUser;
    inputIndices.d[1] = 1;
    inputIndices.d[2] = 1;

    // There should be two input and three output tensors
    assert(mParams.inputTensorNames.size() == 2);
    assert(mParams.outputTensorNames.size() == 3);

    parser->registerInput(mParams.inputTensorNames[0].c_str(), inputIndices, nvuffparser::UffInputOrder::kNCHW);
    parser->registerInput(mParams.inputTensorNames[1].c_str(), inputIndices, nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(mParams.outputTensorNames[0].c_str());

    auto dType = mParams.fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    gLogInfo << "Begin parsing model..." << std::endl;

    // Parse the uff model to populate the network
    if (!parser->parse(mParams.uffFileName.c_str(), *network, dType))
    {
        gLogError << "Failure while parsing UFF file" << std::endl;
        return;
    }

    gLogInfo << "End parsing model..." << std::endl;

    // Add postprocessing i.e. topk layer to the UFF Network
    // Retrieve last layer of UFF Network
    auto uffLastLayer = network->getLayer(network->getNbLayers() - 1);

    // Reshape output of fully connected layer numOfMovies x 1 x 1 x 1 to numOfMovies x 1 x 1.
    auto reshapeLayer = network->addShuffle(*uffLastLayer->getOutput(0));
    reshapeLayer->setReshapeDimensions(nvinfer1::Dims3(1, mParams.numMoviesPerUser, 1));
    assert(reshapeLayer != nullptr);

    // Apply TopK layer to retrieve item probabilities and corresponding index number.
    auto topK = network->addTopK(*reshapeLayer->getOutput(0), nvinfer1::TopKOperation::kMAX, mParams.topKMovies, 0x2);
    assert(topK != nullptr);

    // Mark outputs for index and probs. Also need to set the item layer type == kINT32.
    topK->getOutput(0)->setName(mParams.outputTensorNames[1].c_str());
    topK->getOutput(1)->setName(mParams.outputTensorNames[2].c_str());

    // Specify topK tensors as outputs
    network->markOutput(*topK->getOutput(0));
    network->markOutput(*topK->getOutput(1));

    // Set the topK indices tensor as INT32 type
    topK->getOutput(1)->setType(nvinfer1::DataType::kINT32);

    gLogInfo << "Done constructing network..." << std::endl;

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It
//! allocates the buffer, sets inputs, executes the engine, and verifies the output.
//!
bool SampleMovieLens::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext());

    if (!context)
    {
        return false;
    }

    if (!processInput(buffers))
    {
        return false;
    }

    // Create CUDA stream for the execution of this inference.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    samplesCommon::GpuTimer timer{stream};
    timer.start();

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr))
    {
        return false;
    }

    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);
    timer.stop();
    gLogInfo << "Done execution. Duration : " << timer.microseconds() << " microseconds." << std::endl;

    // Release stream
    cudaStreamDestroy(stream);

    float* topKItemProb = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));
    uint32_t* topKItemNumber = static_cast<uint32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[2]));

    uint32_t* userInput = static_cast<uint32_t*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    uint32_t* itemInput = static_cast<uint32_t*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));

    return SampleMovieLens::verifyOutput(userInput, itemInput, topKItemNumber, topKItemProb);
}

//!
//! \brief Copies a batch of input data from SampleMovieLensParams into managed input buffers
//!
bool SampleMovieLens::processInput(const samplesCommon::BufferManager& buffers)
{
    // Parse ground truth data and inputs
    SampleMovieLens::parseMovieLensData();

    uint32_t* userInput = static_cast<uint32_t*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    uint32_t* itemInput = static_cast<uint32_t*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));

    // Copy batch of inputs to host buffers
    for (int i = 0; i < mParams.batchSize; ++i)
    {
        for (int k = 0; k < mParams.numMoviesPerUser; ++k)
        {
            int idx = i * mParams.numMoviesPerUser + k;
            userInput[idx] = mParams.outParamsVec[i].userId;
            itemInput[idx] = mParams.outParamsVec[i].allItems.at(k);
        }
    }

    return true;
}

//!
//! \brief Helper function to read the next line of the MovieLens dataset
//!        .csv file and return the contents of the line after the delimeter.
//!
//! \details This function is called from SampleMovieLens::readInputSample()
//!          to extract the needed values per user.
std::string SampleMovieLens::readNextLine(std::ifstream& file, char delim)
{
    std::string line;
    std::getline(file, line);
    auto pos = line.find(delim);
    line = line.substr(pos + 1);
    return line;
}

//!
//! \brief Extracts needed dataset values for a single user in the MovieLens,
//!        dataset .csv file, and populates the corresponding ground truth data struct
//!
void SampleMovieLens::readInputSample(std::ifstream& file, OutputParams& outParams, std::string line)
{
    // read user name
    char delim = ':';
    auto pos = line.find(delim);
    line = line.substr(pos + 1);
    outParams.userId = std::stoi(line);
    // read items
    std::string items = readNextLine(file, delim);
    items = items.substr(2, items.size() - 2);
    std::stringstream ss(items);
    std::string i;
    while (ss >> i)
    {
        if (ss.peek() == ',' || ss.peek() == ' ')
        {
            ss.ignore();
        }

        i = i.substr(0, i.size() - 1);
        outParams.allItems.push_back(stoi(i));
    }

    // read expected predicted max rating item
    outParams.expectedPredictedMaxRatingItem = std::stoi(readNextLine(file, delim));

    // read expected predicted max rating prob
    std::string prob = readNextLine(file, delim);
    prob = prob.substr(2, prob.size() - 3);
    outParams.expectedPredictedMaxRatingItemProb = std::stof(prob);

    // skip line
    std::getline(file, line);
    std::getline(file, line);

    // read all the top 10 prediction ratings
    for (int i = 0; i < 10; ++i)
    {
        auto pos = line.find(delim);
        int32_t item = std::stoi(line.substr(0, pos - 1));
        float prob = std::stof(line.substr(pos + 2));
        outParams.itemProbPairVec.emplace_back((make_pair(item, prob)));
        std::getline(file, line);
    }
}

//!
//! \brief Parses the MovieLens dataset and populates the SampleMovieLensParams data structure
//!
void SampleMovieLens::parseMovieLensData()
{
    std::ifstream file;
    file.open(mParams.ratingInputFile, ios::binary);
    std::string line;
    int userIdx = 0;
    while (std::getline(file, line) && userIdx < mParams.batchSize)
    {
        OutputParams outParams;
        readInputSample(file, outParams, line);

        // store the outParams in the class data structure.
        mParams.outParamsVec.push_back(outParams);

        mParams.userToItemsMap[userIdx] = std::move(outParams.allItems);
        mParams.userToExpectedItemProbMap[userIdx] = std::move(outParams.itemProbPairVec);

        userIdx++;
        printOutputParams(outParams);
    }

    // number of users should be equal to number of users in rating file
    assert(mParams.batchSize == userIdx);
}

bool SampleMovieLens::teardown()
{
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

//!
//! \brief Prints the expected recommendation results (ground truth)
//!        from the MovieLens dataset for a given user
//!
void SampleMovieLens::printOutputParams(OutputParams& outParams)
{
    gLogVerbose << "User Id                            :   " << outParams.userId << std::endl;
    gLogVerbose << "Expected Predicted Max Rating Item :   " << outParams.expectedPredictedMaxRatingItem << std::endl;
    gLogVerbose << "Expected Predicted Max Rating Prob :   " << outParams.expectedPredictedMaxRatingItemProb << std::endl;
    gLogVerbose << "Total TopK Items : " << outParams.itemProbPairVec.size() << std::endl;
    for (unsigned int i = 0; i < outParams.itemProbPairVec.size(); ++i)
    {
        gLogVerbose << outParams.itemProbPairVec.at(i).first << " : " << outParams.itemProbPairVec.at(i).second << std::endl;
    }
}

//!
//! \brief Compares the inference output with ground truth and logs the results
//!
bool SampleMovieLens::verifyOutput(uint32_t* userInput, uint32_t* /*itemInput*/, uint32_t* topKItemNumber, float* topKItemProb)
{
    bool pass{true};

    gLogInfo << "Num of users : " << mParams.batchSize << std::endl;
    gLogInfo << "Num of Movies : " << mParams.numMoviesPerUser << std::endl;

    gLogVerbose << "|-----------|------------|-----------------|-----------------|" << std::endl;
    gLogVerbose << "|   User    |   Item     |  Expected Prob  |  Predicted Prob |" << std::endl;
    gLogVerbose << "|-----------|------------|-----------------|-----------------|" << std::endl;

    for (int i = 0; i < mParams.batchSize; ++i)
    {
        int userIdx = userInput[i * mParams.numMoviesPerUser];
        int maxPredictedIdx = topKItemNumber[i * mParams.topKMovies];
        int maxExpectedItem = mParams.userToExpectedItemProbMap.at(userIdx).at(0).first;
        int maxPredictedItem = mParams.userToItemsMap.at(userIdx).at(maxPredictedIdx);
        pass &= maxExpectedItem == maxPredictedItem;

        for (int k = 0; k < mParams.topKMovies; ++k)
        {
            int predictedIdx = topKItemNumber[i * mParams.topKMovies + k];
            float predictedProb = topKItemProb[i * mParams.topKMovies + k];
            float expectedProb = mParams.userToExpectedItemProbMap.at(userIdx).at(k).second;
            int predictedItem = mParams.userToItemsMap.at(userIdx).at(predictedIdx);
            gLogVerbose << "|" << setw(10) << userIdx << " | " << setw(10) << predictedItem << " | " << setw(15) << expectedProb << " | " << setw(15) << predictedProb << " | " << std::endl;
        }
    }

    for (int i = 0; i < mParams.batchSize; ++i)
    {
        int userIdx = userInput[i * mParams.numMoviesPerUser];
        int maxPredictedIdx = topKItemNumber[i * mParams.topKMovies];
        int maxExpectedItem = mParams.userToExpectedItemProbMap.at(userIdx).at(0).first;
        int maxPredictedItem = mParams.userToItemsMap.at(userIdx).at(maxPredictedIdx);
        gLogInfo << "| User :" << setw(4) << userIdx << "  |  Expected Item :" << setw(5) << maxExpectedItem << "  |  Predicted Item :" << setw(5) << maxPredictedItem << " | " << std::endl;
    }

    return pass;
}

struct SampleMovieLensArgs
{
    bool help{false};
    int batchSize{32};
    int dlaCore{-1};
    bool fp16{false};
    bool strict{false};
    bool verbose{false};
};

//!
//! \brief Parses the command line arguments for the MovieLens sample, and returns failure
//!        if arguments are incorrect
//!
bool parseSampleMovieLensArgs(SampleMovieLensArgs& args, int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string argStr(argv[i]);

        if (argStr == "-h" || argStr == "--help")
        {
            args.help = true;
            return true;
        }
        if (argStr == "-b")
        {
            i++;
            args.batchSize = std::atoi(argv[i]);
        }
        else if (argStr == "--fp16")
        {
            args.fp16 = true;
        }
        else if (argStr == "--strict")
        {
            args.strict = true;
        }
        else if (argStr == "--verbose")
        {
            args.verbose = true;
            setReportableSeverity(Logger::Severity::kVERBOSE);
        }
        else if (argStr.substr(0, 13) == "--useDLACore=" && argStr.size() > 13)
        {
            args.dlaCore = stoi(argv[i] + 13);
        }
        else
        {
            return false;
        }
    }
    return true;
}

//!
//! \brief Initializes members of the params struct using the
//!        command line args
//!
SampleMovieLensParams initializeSampleParams(const SampleMovieLensArgs& args)
{
    SampleMovieLensParams params;

    params.dataDirs.push_back("data/movielens/");
    params.dataDirs.push_back("data/samples/movielens/");

    params.uffFileName = locateFile("sampleMovieLens.uff", params.dataDirs);
    params.embeddingVecSize = 32;
    params.topKMovies = 1;
    params.numMoviesPerUser = 100;
    params.ratingInputFile = locateFile("movielens_ratings.txt", params.dataDirs);

    params.inputTensorNames.push_back("user_input");
    params.inputTensorNames.push_back("item_input");
    params.outputTensorNames.push_back("prediction/Sigmoid");
    params.outputTensorNames.push_back("topk_values");
    params.outputTensorNames.push_back("topk_items");

    params.batchSize = args.batchSize;
    params.dlaCore = args.dlaCore;
    params.fp16 = args.fp16;
    params.strict = args.strict;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_movielens [-h or --help] [-b NUM_USERS] [--useDLACore=<int>] [--verbose]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "-b NUM_USERS    Number of Users i.e. Batch Size (default numUsers==32)\n";
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support "
                 "DLA. Value can range from 0 to n-1, where n is the number of "
                 "DLA engines on the platform."
              << std::endl;
    std::cout << "--fp16          Run in FP16 mode.\n";
    std::cout << "--strict        Run with strict type constraints." << std::endl;
}

int main(int argc, char** argv)
{
    SampleMovieLensArgs args;
    bool argsOK = parseSampleMovieLensArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleMovieLensParams params = initializeSampleParams(args);
    SampleMovieLens sample(params);

    gLogInfo << "Building and running a GPU inference engine for MLP NCF model..."
             << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
