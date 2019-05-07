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

//! \file sampleSerialization.cpp
//! \brief This file contains the implementation of the INetwork serialization
//! sample.

#include "NvInfer.h"
#include "NvInferSerialize.h"

#include "NvCaffeParser.h"
#include "NvOnnxParser.h"
#include "NvUffParser.h"

#include "buffers.h"
#include "logger.h"
#include "sampleParseArgs.h"

#include <fstream>
#include <iostream>
#include <sstream>

const std::string gSampleName = "TensorRT.sample_serialization";

const int gBatchSize = 1;
const int gImgChannels = 3;
const int gImgHeight = 224;
const int gImgWidth = 224;

bool operator==(const nvinfer1::Dims& d1, const nvinfer1::Dims& d2)
{
    if (d1.nbDims != d2.nbDims)
        return false;
    for (int i = 0; i < d1.nbDims; ++i)
    {
        if (d1.d[i] != d2.d[i])
            return false;
    }
    return true;
}

//!
//! \brief The SampleParams structure groups the additional parameters
//!     required by the serialization sample.
//!
struct SampleParams
{
    NetworkFormat networkFormat;
    std::vector<std::string> dataDirs;
    std::string caffePrototxtFileName, caffeWeightsFileName;
    std::string onnxFileName;
    std::string uffFileName;
    std::string trtFileName;
    std::string outputPath;
    std::vector<std::string> inputTensorNames, outputTensorNames;
    std::vector<nvinfer1::Dims> inputTensorShapes;
    bool runInference;
};

//!
//! \brief  The SampleSerialization class implements the INetwork
//!     serialization sample.
//!
//! \details It creates the network using provided model and then serializes it
//!     to a TensorRT PLAN file.
//!
class SampleSerialization
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleSerialization(const SampleParams& params)
        : mParams(params)
    {
    }

    //!
    //! \brief This function parses input network.
    //!
    bool parse();

    //!
    //! \brief This function serializes network.
    //!
    bool serialize();

    //!
    //! \brief This function runs inference.
    //!
    bool doInference();

private:
    //!
    //! \brief This function parses Caffe model.
    //!
    bool parseCaffe();

    //!
    //! \brief This function parses ONNX model.
    //!
    bool parseONNX();

    //!
    //! \brief This function parses UFF model.
    //!
    bool parseUff();

    //!
    //! \brief This function deserializes TensorRT PLAN file.
    //!
    bool parseTRT();

    SampleParams mParams; //!< The parameters for the sample.

    SampleUniquePtr<nvinfer1::IBuilder> mBuilder;           //!< The builder for the network
    SampleUniquePtr<nvinfer1::INetworkDefinition> mNetwork; //!< Parsed network
    SampleUniquePtr<nvinfer1::INetworkConfig> mConfig;  //!< Network configuration

    SampleUniquePtr<nvonnxparser::IParser> mOnnxParser;
    SampleUniquePtr<nvuffparser::IUffParser> mUffParser;
    SampleUniquePtr<nvcaffeparser1::ICaffeParser> mCaffeParser;
    SampleUniquePtr<nvinfer1::serialize::IDeserializer> mDeserializer;
};

bool SampleSerialization::parse()
{
    mBuilder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!mBuilder)
        return false;

    mNetwork = SampleUniquePtr<nvinfer1::INetworkDefinition>(mBuilder->createNetwork());
    if (!mNetwork)
        return false;

    mConfig = SampleUniquePtr<nvinfer1::INetworkConfig>(mBuilder->createNetworkConfig());
    if (!mConfig)
        return false;

    switch (mParams.networkFormat)
    {
    case NetworkFormat::kCAFFE:
        return parseCaffe();
    case NetworkFormat::kONNX:
        return parseONNX();
    case NetworkFormat::kUFF:
        return parseUff();
    case NetworkFormat::kTRT:
        return parseTRT();
    }

    return true;
}

bool SampleSerialization::serialize()
{
    gLogInfo << "Serializing parsed network to " << mParams.outputPath << std::endl;
    auto serializedNetwork = SampleUniquePtr<nvinfer1::IHostMemory>(
        nvinfer1::serialize::serializeNetwork(mNetwork.get()));

    std::ofstream output(mParams.outputPath, std::ios::binary);
    if (output.is_open())
    {
        output.write(reinterpret_cast<const char*>(serializedNetwork->data()),
                     serializedNetwork->size());
        output.close();
        return true;
    }
    return false;
}

bool SampleSerialization::parseCaffe()
{
    gLogInfo << "Parsing Caffe network from " << mParams.caffePrototxtFileName
             << " and " << mParams.caffeWeightsFileName << std::endl;
    mCaffeParser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(
        nvcaffeparser1::createCaffeParser());
    if (!mCaffeParser)
        return false;

    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = mCaffeParser->parse(
        locateFile(mParams.caffePrototxtFileName, mParams.dataDirs).c_str(),
        locateFile(mParams.caffeWeightsFileName, mParams.dataDirs).c_str(),
        *mNetwork, nvinfer1::DataType::kFLOAT);

    for (auto& s : mParams.outputTensorNames)
        mNetwork->markOutput(*blobNameToTensor->find(s.c_str()));

    return true;
}

bool SampleSerialization::parseONNX()
{
    gLogInfo << "Parsing ONNX network from " << mParams.onnxFileName << std::endl;
    mOnnxParser = SampleUniquePtr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*mNetwork, gLogger.getTRTLogger()));
    if (!mOnnxParser)
        return false;

    std::string file = locateFile(mParams.onnxFileName, mParams.dataDirs);
    if (!mOnnxParser->parseFromFile(file.c_str(), 0))
        return false;

    return true;
}

bool SampleSerialization::parseUff()
{
    gLogInfo << "Parsing UFF network from " << mParams.uffFileName << std::endl;
    mUffParser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if (!mUffParser)
        return false;

    for (unsigned i = 0; i < mParams.inputTensorNames.size(); ++i)
        mUffParser->registerInput(mParams.inputTensorNames[i].c_str(),
                                  mParams.inputTensorShapes[i],
                                  nvuffparser::UffInputOrder::kNCHW);

    std::string file = locateFile(mParams.uffFileName, mParams.dataDirs);
    if (!mUffParser->parse(file.c_str(), *mNetwork))
        return false;

    return true;
}

bool SampleSerialization::parseTRT()
{
    gLogInfo << "Deserializing TensorRT network from " << mParams.trtFileName << std::endl;
    mDeserializer = SampleUniquePtr<nvinfer1::serialize::IDeserializer>(
        nvinfer1::serialize::createDeserializer(mNetwork.get(), getLogger()));

    std::vector<char> modelStream;
    size_t size{0};
    std::string fileName = locateFile(mParams.trtFileName, mParams.dataDirs);
    std::ifstream file(fileName, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        modelStream.resize(size);
        file.read(modelStream.data(), size);
        file.close();

        bool good = mDeserializer->deserialize(modelStream.data(), size);
        return good;
    }
    gLogInfo << "Couldn\'t read from " << mParams.trtFileName << std::endl;
    return false;
}

bool SampleSerialization::doInference()
{
    const nvinfer1::Dims inputDims{3, {gImgChannels, gImgHeight, gImgWidth}};
    assert(mNetwork->getNbInputs() == 1);
    assert(mNetwork->getInput(0)->getDimensions() == inputDims);
    assert(mNetwork->getNbOutputs() == 1);
    const nvinfer1::Dims outputDims = mNetwork->getOutput(0)->getDimensions();
    assert(outputDims.nbDims >= 1);
    for (int i = 1; i < outputDims.nbDims; ++i)
        assert(outputDims.d[i] == 1);
    int nbClasses = outputDims.d[0];

    // Allocate memory for input and output tensors
    std::vector<float> inputData(gBatchSize * gImgChannels * gImgHeight * gImgWidth);
    std::vector<float> predictions(gBatchSize * nbClasses);

    // Read input image and load it into input buffer
    samplesCommon::PPM<gImgChannels, 224, 226> ppm;
    std::string imagePath = locateFile("image.ppm", mParams.dataDirs);
    samplesCommon::readPPMFile(imagePath, ppm);
    for (int i = 0, volImg = gImgChannels * gImgHeight * gImgWidth; i < gBatchSize; ++i)
    {
        for (int c = 0; c < gImgChannels; ++c)
        {
            // The color image to input should be in RGB order
            for (unsigned j = 0, volChl = gImgHeight * gImgWidth; j < volChl; ++j)
            {
                inputData[i * volImg + c * volChl + j] = ppm.buffer[j * gImgChannels + c];
            }
        }
    }

    mBuilder->setMaxBatchSize(gBatchSize);
    mConfig->setMaxWorkspaceSize(10_MB);

    // Build engine and execution context from network
    gLogInfo << "Building engine" << std::endl;
    auto engine = SampleUniquePtr<nvinfer1::ICudaEngine>(mBuilder->buildEngineWithConfig(*mNetwork, *mConfig));
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

    assert(engine->getNbBindings() == 2);
    void* buffers[2];

    int inputIndex = engine->getBindingIndex(mNetwork->getInput(0)->getName()),
        outputIndex = engine->getBindingIndex(mNetwork->getOutput(0)->getName());

    CHECK(cudaMalloc(&buffers[inputIndex], gBatchSize * gImgChannels * gImgHeight * gImgWidth * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], gBatchSize * nbClasses * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    gLogInfo << "Running inference" << std::endl;
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData.data(), gBatchSize * gImgChannels * gImgHeight * gImgWidth * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(gBatchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(predictions.data(), buffers[outputIndex], gBatchSize * nbClasses * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

    // Find the best prediction for each input image
    for (int n = 0; n < gBatchSize; ++n)
    {
        int bestIndex = 0;
        float bestValue = predictions[n * nbClasses];
        for (int i = 0; i < nbClasses; ++i)
        {
            if (predictions[n * nbClasses + i] > bestValue)
            {
                bestIndex = i;
                bestValue = predictions[n * nbClasses + i];
            }
        }
        gLogInfo << "Best index = " << bestIndex << ", prob = " << bestValue << std::endl;
    }

    return true;
}

//!
//! \brief This function initializes members of the params struct using the
//!     command line args.
//!
SampleParams initializeSampleParams(const SampleArgs& args)
{
    SampleParams params;
    if (!args.dataDirs.empty()) //!< Use the data directory provided by the user
        params.dataDirs = args.dataDirs;
    else //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/ResNet50/");
    }
    params.networkFormat = args.networkFormat;
    params.caffePrototxtFileName = args.caffePrototxtFileName;
    params.caffeWeightsFileName = args.caffeWeightsFileName;
    params.onnxFileName = args.onnxFileName;
    params.uffFileName = args.uffFileName;
    params.trtFileName = args.trtFileName;

    if (args.inputTensorNames.size() != 0)
        params.inputTensorNames = args.inputTensorNames;
    else
    {
        params.inputTensorNames.push_back("data");
    }
    if (args.outputTensorNames.size() != 0)
        params.outputTensorNames = args.outputTensorNames;
    else
    {
        params.outputTensorNames.push_back("prob");
    }

    for (const std::vector<int>& shape : args.inputTensorShapes)
    {
        nvinfer1::Dims dims;
        dims.nbDims = shape.size() - 1;
        for (unsigned dim = shape.size() - 1; dim > 0; --dim)
        {
            dims.d[dim - 1] = shape[dim];
        }

        params.inputTensorShapes.push_back(dims);
    }

    params.outputPath = args.outputPath;
    params.runInference = args.runInference;

    return params;
}

//!
//! \brief This function prints the help information for running this sample.
//!
void printHelpInfo()
{
    std::cout << "Possible arguments:" << std::endl;
    std::cout << "-h/--help        Display help information." << std::endl;
    std::cout << "-d/--datadir     Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/ResNet50/)." << std::endl;
    std::cout << "--format         Specify format of input files, should be one of: caffe, onnx, uff, trt. Default is \"caffe\"." << std::endl;
    std::cout << "--caffe_prototxt Path to caffe .prototxt file. Default is \"ResNet50_N2.prototxt\"." << std::endl;
    std::cout << "--caffe_weights  Path to caffe .caffemodel file. Default is \"ResNet50_fp32.caffemodel\"." << std::endl;
    std::cout << "--onnx_file      Path to ONNX .onnx file." << std::endl;
    std::cout << "--uff_file       Path to UFF .uff file." << std::endl;
    std::cout << "--trt_file       Path to serialized tnb file." << std::endl;
    std::cout << "--run_inference  Run example inference on parsed/deserialized network. Inference assumes that network takes one input of 3,224,224 shape (ImageNet picture) and returns vector of probabilities." << std::endl;
    std::cout << "--input_name     Specify name of input tensor. This option can be used multiple times to add multiple input tensors. Default is (\"data\")." << std::endl;
    std::cout << "--output_name    Specify name of output tensor. This option can be used multiple times to add multiple output tensors. Default is (\"prob\")." << std::endl;
    std::cout << "--input_shape    The shape of input tensor, specified as a comma separated list of numbers. This option can be used multiple times to add multiple input shapes. For more detailed explanation please refer to README.md." << std::endl;
    std::cout << "-o/--output      Path to where serialized network will be saved." << std::endl;
}

int main(int argc, char** argv)
{
    SampleArgs args;

    bool argsOK = parseArgs(argc, argv, args);

    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        return EXIT_FAILURE;
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    SampleParams params = initializeSampleParams(args);

    SampleSerialization sample(params);

    if (!sample.parse())
    {
        return gLogger.reportFail(sampleTest);
    }

    if (params.networkFormat != NetworkFormat::kTRT && !sample.serialize())
    {
        return gLogger.reportFail(sampleTest);
    }

    if (params.runInference && !sample.doInference())
    {
        return gLogger.reportFail(sampleTest);
    }

    return gLogger.reportPass(sampleTest);
}
