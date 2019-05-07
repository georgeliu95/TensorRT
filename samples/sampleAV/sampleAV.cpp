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

#include "BatchStream.h"
#include "InternalAPI.h"
#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvUffParser.h"
#include "common.h"
#include "logger.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime_api.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unordered_map>

using namespace nvinfer1;
using namespace nvcaffeparser1;

const std::string gSampleName = "TensorRT.sample_av";

static int gUseDLACore = -1;
static int gAvgRuns{100};
static std::string gInputPath = ".";  // default input data directory
static std::string gOutputPath = "."; // default output directory - stores output dump and calibration file
static std::string gNetworkName = "drivenet_GridboxModel";
static int gNbCalibBatches = 100;
static int gCalibBatchSize = 1;
static int gNbInferBatches = 25;
static int gInferBatchSize = 16;
static float gInt8Tolerence = 0.27f;
static float gFp16Tolerence = 0.27f;
static int gDataPointCount = 0;

static int gBackendStart = 0;
static int gBackendStop = INT_MAX;
static int gBackendSize = INT_MAX;

static bool gRunFp16 = false;
static bool gRunInt8 = false;
static bool gUffInput = false;

static bool gVerbose = false;

static std::vector<vector<float>> gInt8Data = {};
static std::vector<vector<float>> gFp32Data = {};
static std::vector<vector<float>> gFp16Data = {};

std::vector<std::string> gDirs;

void setupDirs()
{
    if (!gInputPath.empty())
    {
        gDirs.push_back(gInputPath + "/" + gNetworkName + std::string("/"));
    }
    gDirs.push_back(std::string("int8/AV/") + gNetworkName + std::string("/"));
    gDirs.push_back(std::string("data/AV/") + gNetworkName + std::string("/"));
    gDirs.push_back(std::string("data/int8/AV/") + gNetworkName + std::string("/"));
    gDirs.push_back(std::string("data/int8_samples/AV/") + gNetworkName + std::string("/"));
    gDirs.push_back(std::string("data/samples/AV/") + gNetworkName + std::string("/"));
    if (gUffInput)
    {
        gDirs.push_back(std::string("data/uff_data/new_uffs/") + gNetworkName + std::string("/"));
    }
}

std::string locateInputFile(const std::string& input)
{
    return locateFile(input, gDirs);
}

std::string locateOutputFile(const std::string& output)
{
    return gOutputPath + "/" + output;
}

//!
//! \brief Struct of input blob
//!
struct InputBlobInfo
{
    InputBlobInfo(const char* blob_name)
        : name(blob_name)
    {
    }
    std::string name;
    int index;
    int size;
    DimsCHW dim;
};

//!
//! \brief Struct of output blob
//!
struct OutputBlobInfo
{
    OutputBlobInfo(const char* blob_name, const char* blob_type)
        : name(blob_name)
        , type(blob_type)
    {
    }
    std::string name;
    std::string type;
    int index;
    int size;
    DimsCHW dim;
    float* cpu_buff;
};

class Int8EntropyCalibrator2 : public IInt8EntropyCalibrator2
{
public:
    // constructor
    Int8EntropyCalibrator2(BatchStream& stream, int firstBatch, const std::string& input_name, bool readCache = true)
        : mStream(stream)
        , mReadCache(readCache)
        , mInputName(input_name)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    // destructor
    virtual ~Int8EntropyCalibrator2()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    // get batch size
    int getBatchSize() const override { return mStream.getBatchSize(); }

    // read batch
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
        {
            return false;
        }

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], mInputName.c_str()));
        bindings[0] = mDeviceInput;
        return true;
    }

    // read calibration cache
    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();

        // First, look for calibration table in the input path.
        std::ifstream ctFileStream(locateInputFile("CalibrationTable_" + gNetworkName), std::ios::binary);
        if (ctFileStream)
        {
            ctFileStream >> std::noskipws;
            if (mReadCache && ctFileStream.good())
            {
                std::copy(std::istream_iterator<char>(ctFileStream), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
            }
            length = mCalibrationCache.size();
            return length ? &mCalibrationCache[0] : nullptr;
        }

        // If calibration table was not found at input path, look for calibration cache from the previous runs.
        std::ifstream ctCacheStream(locateOutputFile("CalibrationTable_" + gNetworkName), std::ios::binary);
        if (ctCacheStream)
        {
            ctCacheStream >> std::noskipws;
            if (mReadCache && ctCacheStream.good())
            {
                std::copy(std::istream_iterator<char>(ctCacheStream), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
            }

            length = mCalibrationCache.size();
            return length ? &mCalibrationCache[0] : nullptr;
        }

        return nullptr;
    }

    // write calibration cache
    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(locateOutputFile("CalibrationTable_" + gNetworkName), std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    BatchStream mStream;
    bool mReadCache{true};
    size_t mInputCount;
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
    std::string mInputName;
};

//!
//! \brief This function returns Get CHW element number for dim.
//!
int getCHWSize(const DimsCHW dim)
{
    return dim.c() * dim.h() * dim.w();
}

//!
//! \brief This function returns Caffe model filename for the network.
//!
std::string caffeModelName(const std::string& gNetworkName)
{
    return gNetworkName + ".caffemodel";
}

//!
//! \brief This function returns Caffe prototxt filename for the network.
//!
std::string caffePrototxtName(const std::string& gNetworkName)
{
    return gNetworkName + ".prototxt";
}

//!
//! \brief This function returns uff filename for the network.
//!
std::string uffName(const std::string& gNetworkName)
{
    return gNetworkName + ".uff";
}

//!
//! \brief Configure builder
//!
void configureBuilder(IBuilder* builder,
                      INetworkConfig * config,
                      unsigned int maxBatchSize,
                      bool fp16Mode,
                      bool int8Mode,
                      IInt8Calibrator* calibrator)
{
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1_GB);
    config->setAvgTimingIterations(1);
    config->setMinTimingIterations(1);
    config->setFlag(BuilderFlag::kDEBUG);

    if (calibrator && int8Mode)
    {
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator);
        samplesCommon::enableDLA(builder, config, gUseDLACore);
    }
    else if (fp16Mode)
    {
        builder->setFp16Mode(true);
        config->setFlag(BuilderFlag::kFP16);
        samplesCommon::enableDLA(builder, config, gUseDLACore);
    }

    setBackendNodeRange(*builder, gBackendStart, gBackendStop, gBackendSize);
}

//!
//! \brief This function uses a UFF parser to parse the network.
//!
ICudaEngine* uffToTRTModel(IBuilder* builder,
                           INetworkDefinition* network,
                           INetworkConfig* config,
                           const std::vector<struct InputBlobInfo>& uffInputs,  // network inputs
                           const std::vector<struct OutputBlobInfo>& uffOutputs) // network outputs
{
    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();

    // specify which tensors are outputs
    for (auto& output : uffOutputs)
    {
        if (!parser->registerOutput(output.name.c_str()))
        {
            gLogError << "Failed to register output " << output.name << std::endl;
            return nullptr;
        }
    }

    // specify which tensors are inputs (and their dimensions)
    for (auto& input : uffInputs)
    {
        if (!parser->registerInput(input.name.c_str(), input.dim, nvuffparser::UffInputOrder::kNCHW))
        {
            gLogError << "Failed to register input " << input.name << std::endl;
            return nullptr;;
        }
    }

    std::string uffFile{locateInputFile(uffName(gNetworkName))};
    if (uffFile.empty())
    {
        exit(1);
    }

    if (!parser->parse(uffFile.c_str(), *network))
    {
        parser->destroy();
        return nullptr;
    }

    // Build the engine
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);

    parser->destroy();

    return engine;
}

//!
//! \brief This function uses a Caffe parser to parse the network
//!
ICudaEngine* caffeToTRTModel(IBuilder* builder,
                             INetworkDefinition* network,
                             INetworkConfig* config,
                             const std::vector<struct OutputBlobInfo>& outputs) // network outputs
{
    ICaffeParser* parser = createCaffeParser();
    std::string prototxtFile{locateInputFile(caffePrototxtName(gNetworkName))};
    if (prototxtFile.empty())
    {
        exit(1);
    }
    std::string modelFile{locateInputFile(caffeModelName(gNetworkName))};
    if (modelFile.empty())
    {
        exit(1);
    }

    const IBlobNameToTensor* blobNameToTensor = parser->parse(prototxtFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,
                                                              DataType::kFLOAT);

    // specify which tensors are outputs
    for (auto& s : outputs)
    {
        network->markOutput(*blobNameToTensor->find(s.name.c_str()));
    }

    // Build the engine
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    assert(engine);

    parser->destroy();
    return engine;
}

//!
//! \brief This function creates TRT engine.
//!
ICudaEngine* createTRTEngine(const std::vector<struct InputBlobInfo>& inputs,   // network inputs
                             const std::vector<struct OutputBlobInfo>& outputs, // network outputs
                             unsigned int maxBatchSize,                         // batch size
                             bool fp16Mode,
                             bool int8Mode)
{
    std::unique_ptr<IInt8Calibrator> calibrator;
    if (int8Mode)
    {
        // define calibration stream
        BatchStream calibrationStream(gCalibBatchSize, gNbCalibBatches, std::string("infer_batches/batch"), gDirs);
        gLogInfo << "Using Entropy Calibrator 2" << std::endl;
        // define calibrator
        calibrator.reset(new Int8EntropyCalibrator2(calibrationStream,
                                                    0,
                                                    inputs[0].name));
    }

    // create builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    INetworkConfig* config = builder->createNetworkConfig();
    configureBuilder(builder, config, maxBatchSize, fp16Mode, int8Mode, calibrator.get());

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();

    ICudaEngine* engine{nullptr};

    if (gUffInput)
    {
        engine = uffToTRTModel(builder, network, config, inputs, outputs);
    }
    else 
    {
        engine = caffeToTRTModel(builder, network, config, outputs);
    }

    assert(engine);
    // destroy network, parser and builder
    network->destroy();
    builder->destroy();
    config->destroy();

    calibrator.reset();

    return engine;
}

//!
//! \brief This function does inference for one frame.
//!
float doInference(IExecutionContext& context, float* input,
                  const int& input_num, const int& output_num,
                  struct InputBlobInfo in_blob,
                  std::vector<struct OutputBlobInfo>& out_blobs,
                  DimsCHW inputDims)
{
    const ICudaEngine& engine = context.getEngine();

    // input and output buffer pointers passed to the engine
    assert(engine.getNbBindings() == input_num + output_num);
    void* buffers[input_num + output_num];

    // create GPU buffers
    in_blob.size = gInferBatchSize * getCHWSize(inputDims) * sizeof(float);

    CHECK(cudaMalloc(&buffers[in_blob.index], in_blob.size));

    for (auto& item : out_blobs)
    {
        item.size = getCHWSize(item.dim) * gInferBatchSize * sizeof(float);
        CHECK(cudaMalloc(&buffers[item.index], item.size));
    }

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpy(buffers[in_blob.index], input, in_blob.size, cudaMemcpyHostToDevice));

    // execute inference
    float totalTimeMs{0.0f};
    float iterationTimeMs{0.0f};
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    for (int i = 0; i < gAvgRuns; i++)
    {
        cudaEventRecord(start, stream);
        context.enqueue(gInferBatchSize, buffers, stream, nullptr);
        cudaEventRecord(end, stream);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&iterationTimeMs, start, end);
        totalTimeMs += iterationTimeMs;
    }
    totalTimeMs /= gAvgRuns;

    // copy output from GPU buffers to CPU buffers
    for (auto& item : out_blobs)
    {
        CHECK(cudaMemcpy(item.cpu_buff, buffers[item.index], item.size, cudaMemcpyDeviceToHost));
    }

    // release the stream and the buffers
    CHECK(cudaFree(buffers[in_blob.index]));
    for (auto& item : out_blobs)
    {
        CHECK(cudaFree(buffers[item.index]));
    }

    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return totalTimeMs;
}

//!
//! \brief This function processes inference output.
//!
std::vector<std::vector<float>> processOutput(std::vector<struct OutputBlobInfo>& out_blobs,
                                              const int& idx, std::vector<std::string> output_files)
{
    std::vector<std::vector<float>> outs;
    int bcount = 0;
    for (auto& item : out_blobs)
    {
        std::vector<float> dataPoints;
        std::ofstream of(output_files[bcount], std::ofstream::out | std::ofstream::app);
        std::stringstream ss_data;
        for (int i = 0; i < gInferBatchSize * getCHWSize(item.dim); i++)
        {
            ss_data << ++gDataPointCount << ": " << item.cpu_buff[i] << std::endl;
            dataPoints.emplace_back(item.cpu_buff[i]);
        }
        outs.emplace_back(dataPoints);
        of << ss_data.str();
        of.close();
        bcount++;
    }
    return outs;
}

//!
//! \brief This function runs inference on all frames.
//!
void scoreNetwork(int nbBatches,
                  const int& input_num,
                  const int& output_num,
                  std::vector<struct InputBlobInfo>& inputs,
                  std::vector<struct OutputBlobInfo>& outputs,
                  bool fp16Mode,
                  bool int8Mode)
{
    ICudaEngine* engine = createTRTEngine(inputs, outputs, gInferBatchSize, fp16Mode, int8Mode);

    // run end to end inference multiple times
    IExecutionContext* context = engine->createExecutionContext();

    context->setDebugSync(true);
    BatchStream infer_stream(gInferBatchSize, nbBatches, std::string("infer_batches/batch"), gDirs);

    std::string suffix = int8Mode ? "_INT8" : fp16Mode ? "_FP16" : "_FP32";

    std::vector<std::stringstream> fs(outputs.size());
    std::vector<std::string> output_files;
    for (size_t i = 0; i < outputs.size(); ++i)
    {
        fs[i] << gNetworkName << "_" << outputs[i].type << suffix << ".txt";
        std::string file = locateOutputFile(fs[i].str());
        std::remove(file.c_str()); // Remove file if already exists.
        output_files.emplace_back(file);
        gLogInfo << "Writing output to a file: " << file << std::endl;
    }

    std::vector<std::vector<float>>& dataPointsAgg = int8Mode ? gInt8Data : fp16Mode ? gFp16Data : gFp32Data;
    gDataPointCount = 0;

    int idx = 0;
    float totalTime = 0.0f;
    struct InputBlobInfo& input = inputs[0];
    while (infer_stream.next())
    {
        // get dimention of tensors
        // get engine
        const ICudaEngine& engine = context->getEngine();

        // create CPU buffers for outputs
        input.index = engine.getBindingIndex(input.name.c_str());
        DimsCHW inputDims = static_cast<DimsCHW&&>(engine.getBindingDimensions(input.index));

        for (auto& item : outputs)
        {
            item.index = engine.getBindingIndex(item.name.c_str());
            item.dim = static_cast<DimsCHW&&>(engine.getBindingDimensions((int) item.index));
            item.cpu_buff = reinterpret_cast<float*>(malloc(gInferBatchSize * getCHWSize(item.dim) * sizeof(float)));
        }

        // do inference
        totalTime += doInference(*context, infer_stream.getBatch(), input_num, output_num, input, outputs, inputDims);

        // Prepare inference output.
        auto dp = processOutput(outputs, idx, output_files);
        dataPointsAgg.insert(dataPointsAgg.end(), dp.begin(), dp.end());

        // release CPU output buffers
        for (auto& item : outputs)
        {
            free(item.cpu_buff);
        }

        idx++;
    }

    // release resources
    context->destroy();
    engine->destroy();

    int imagesRead = infer_stream.getBatchesRead() * gInferBatchSize;
    gLogInfo << "Processing " << imagesRead << " images averaged " << totalTime / imagesRead << " ms/image and " << totalTime / infer_stream.getBatchesRead() << " ms/batch." << std::endl;
}

//!
//! \brief This function prints help information of sample_av.
//!
void printUsage()
{
    std::cout << " Usage:" << std::endl;
    std::cout << " sample_av" << std::endl;
    std::cout << " --network_name=network_name. Network name e.g. drivenet_GridBoxModel. Corresponding prototxt and weights files should be drivenet_GridBoxModel.prototxt and drivenet_GridBoxModel.caffemodel respectively." << std::endl;
    std::cout << " --num_calib_batches=N. Number of calibration batches. " << std::endl;
    std::cout << " --calib_batch_size=N. Calibration batch size." << std::endl;
    std::cout << " --num_infer_batches=N. Number of inference batches." << std::endl;
    std::cout << " --infer_batch_size=N. Inference batch size." << std::endl;
    std::cout << " --input_path=/path/to/input/data/dir. Input path to network level directory. The prototxt and weights file paths will be resolved as <input_path>/<network_name>/*." << std::endl;
    std::cout << " --ouput_path=/path/to/output/data/dir. Output path to store fp32, fp16, int8 blob output. Calibration table will also be stored in output path." << std::endl;
    std::cout << " --useDLACore=N. Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << " --int8_tolerance=N.f. Int8 toleranace used to compare against fp32 blob output." << std::endl;
    std::cout << " --fp16_tolerance=N.f. Fp16 tolerance used to compare against fp32 blob output." << std::endl;
    std::cout << " --backend_start=N. Start adding nodes to the backend from N inclusive. Defaults to 0." << std::endl;
    std::cout << " --backend_stop=N. Stop adding nodes to the backend at N inclusive. Defaults to INT_MAX." << std::endl;
    std::cout << " --backend_size=N. Sets the maximum number of nodes a backend can contain. Defaults to INT_MAX." << std::endl;
    std::cout << " --fp16. Run only Fp16 accuracy tests." << std::endl;
    std::cout << " --int8. Run only Int8 accuracy tests." << std::endl;
    std::cout << " --uff. UFF input." << std::endl;
    std::cout << " --verbose. Use verbose logging (default = false)" << std::endl;

    exit(1);
}

//!
//! \brief This function parses the command line arguments
//!
void parseArgs(int argc, char** argv)
{
    struct stat info;
    for (int i = 1; i < argc; i++)
    {
        if (!strncmp(argv[i], "--network_name=", 15))
        {
            gNetworkName = argv[i] + 15;
        }
        else if (!strncmp(argv[i], "--num_calib_batches=", 20))
        {
            gNbCalibBatches = atoi(argv[i] + 20);
        }
        else if (!strncmp(argv[i], "--calib_batch_size=", 19))
        {
            gCalibBatchSize = atoi(argv[i] + 19);
        }
        else if (!strncmp(argv[i], "--num_infer_batches=", 20))
        {
            gNbInferBatches = atoi(argv[i] + 20);
        }
        else if (!strncmp(argv[i], "--infer_batch_size=", 19))
        {
            gInferBatchSize = atoi(argv[i] + 19);
        }
        else if (!strncmp(argv[i], "--int8_tolerance=", 17))
        {
            gInt8Tolerence = atof(argv[i] + 17);
        }
        else if (!strncmp(argv[i], "--fp16_tolerance=", 17))
        {
            gFp16Tolerence = atof(argv[i] + 17);
        }
        else if (!strncmp(argv[i], "--input_path=", 13))
        {
            gInputPath = std::string(argv[i] + 13);
            if (stat(gInputPath.c_str(), &info) != 0)
            {
                gLogInfo << "cannot access input_path " << gInputPath << std::endl;
                exit(1);
            }
        }
        else if (!strncmp(argv[i], "--output_path=", 14))
        {
            gOutputPath = std::string(argv[i] + 14);
            if (stat(gOutputPath.c_str(), &info) != 0)
            {
                gLogInfo << "cannot access output_path " << gOutputPath << std::endl;
                exit(1);
            }
        }
        else if (!strncmp(argv[i], "--useDLACore=", 13))
        {
            gUseDLACore = atoi(argv[i] + 13);
        }
        else if (!strncmp(argv[i], "--avg_runs=", 11))
        {
            gAvgRuns = stoi(argv[i] + 11);
        }
        else if (!strncmp(argv[i], "--backend_start=", 16))
        {
            gBackendStart = stoi(argv[i] + 16);
        }
        else if (!strncmp(argv[i], "--backend_stop=", 15))
        {
            gBackendStop = stoi(argv[i] + 15);
        }
        else if (!strncmp(argv[i], "--backend_size=", 15))
        {
            gBackendSize = stoi(argv[i] + 15);
        }
        else if (!strncmp(argv[i], "--backend_size=", 15))
        {
            gBackendSize = stoi(argv[i] + 15);
        }
        else if (!strncmp(argv[i], "--fp16", 6))
        {
            gRunFp16 = true;
        }
        else if (!strncmp(argv[i], "--int8", 6))
        {
            gRunInt8 = true;
        }
        else if (!strncmp(argv[i], "--uff", 5))
        {
            gUffInput = true;
        } 
        else if (!strncmp(argv[i], "--verbose", 9))
        {
            gVerbose = true;
        } 
        else if (!strncmp(argv[i], "-h", 2) || !(strncmp(argv[i], "--help", 6)))
        {
            printUsage();
        }
        else
        {
            printUsage();
        }
    }
    //! Setup directories
    setupDirs();
}

bool diff(float ref, float val, float tolerance)
{
    if (ref > std::numeric_limits<double>::epsilon() && val > std::numeric_limits<double>::epsilon())
    {
        return std::fabs(ref - val) <= tolerance * std::max(fabs(ref), 1.0f);
    }
    return true;
}

void compareOutputs(std::vector<struct OutputBlobInfo>& outputs)
{
    int blobSize = outputs.size();
    std::vector<float> blobTotal(blobSize, 0);
    for (size_t i = 0; i < gFp32Data.size(); ++i)
    {
        blobTotal[i % blobSize] += gFp32Data[i].size();
    }

    if (gRunFp16)
    {
        assert(gFp32Data.size() == gFp16Data.size());
        std::vector<float> blobFp16Acc(blobSize, 0);

        for (size_t i = 0; i < gFp32Data.size(); ++i)
        {
            int fp16Acc = 0;
            for (size_t j = 0; j < gFp32Data[i].size(); ++j)
            {
                fp16Acc += diff(gFp32Data[i][j], gFp16Data[i][j], gFp16Tolerence);
            }
            blobFp16Acc[i % blobSize] += fp16Acc;
        }

        for (int i = 0; i < blobSize; ++i)
        {
            float fp16Score = ((float) blobFp16Acc[i] / blobTotal[i]) * 100;
            gLogInfo << outputs[i].name << "[" << blobTotal[i] << "]: Fp16 Inference Match Percentage wrt. to Fp32 blob with tolerance [" << gFp16Tolerence << ".f] : " << fp16Score << " %" << std::endl;
        }
    }

    if (gRunInt8)
    {
        assert(gFp32Data.size() == gInt8Data.size());
        std::vector<float> blobInt8Acc(blobSize, 0);

        for (size_t i = 0; i < gFp32Data.size(); ++i)
        {
            int int8Acc = 0;
            assert(gFp32Data[i].size() == gInt8Data[i].size());
            for (size_t j = 0; j < gFp32Data[i].size(); ++j)
            {
                int8Acc += diff(gFp32Data[i][j], gInt8Data[i][j], gInt8Tolerence);
            }
            blobInt8Acc[i % blobSize] += int8Acc;
        }

        for (int i = 0; i < blobSize; ++i)
        {
            float int8Score = ((float) blobInt8Acc[i] / blobTotal[i]) * 100;
            gLogInfo << outputs[i].name << "[" << blobTotal[i] << "]: Int8 Inference Match Percentage wrt. to Fp32 blob with tolerance [" << gInt8Tolerence << ".f] : " << int8Score << " %" << std::endl;
        }

    }

}

int main(int argc, char** argv)
{
    // Parse arguments
    parseArgs(argc, argv);

    if (gVerbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }
    // If the user hasn't specified any of the two runFp16 and runInt8 options, run both Fp16 and Int8 by default.
    if (!gRunFp16 && !gRunInt8)
    {
        gRunFp16 = true;
        gRunInt8 = true;
    }

    // Define model info vars
    int input_num = 1;
    int output_num = -1;

    std::vector<struct InputBlobInfo> inputs = {InputBlobInfo("data")};

    std::vector<struct OutputBlobInfo> outputs;

    if (gNetworkName == "OpenRoadNet_v6")
    {
        gFp16Tolerence = 0.3f;
        gInt8Tolerence = 0.3f;
        inputs[0].name = "input_images";
        inputs[0].dim = DimsCHW(3,272,480);
        output_num = 2;
        outputs = {OutputBlobInfo("boundary_preds/BiasAdd", "boundary_preds_BiasAdd"),
                   OutputBlobInfo("label_preds/BiasAdd", "label_preds_BiasAdd")};
    }
    else if (gNetworkName == "drivenet_GridboxModel")
    {
        output_num = 3;
        inputs[0].name = "input_1";
        outputs = {OutputBlobInfo("output_cov/Sigmoid", "output_cov_Sigmoid"),
                   OutputBlobInfo("output_bbox", "output_bbox"),
                   OutputBlobInfo("output_orientation/Tanh", "output_orientation_Tanh")};
    }
    else if (gNetworkName == "mapnet_v1_8class" || gNetworkName == "MapNet_v1_9")
    {
        gFp16Tolerence = 0.2f;
        gInt8Tolerence = 0.2f;
        output_num = 1;
        inputs[0].name = "input_1";
        inputs[0].dim = DimsCHW(3,240,480);
        outputs = {OutputBlobInfo("activation_54/Sigmoid", "activation_54_Sigmoid")};
    }
    else
    {
        gLogInfo << "Network not supported." << std::endl;
        exit(1);
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);
    gLogger.reportTestStart(sampleTest);

    gLogInfo << "FP32 run:" << gNbInferBatches << " batches of size " << gInferBatchSize << std::endl;
    scoreNetwork(gNbInferBatches, input_num, output_num, inputs, outputs, false /*fp16Mode*/, false /*int8Mode*/);
    gLogInfo << std::endl;

    if (gRunFp16)
    {
        gLogInfo << "FP16 run:" << gNbInferBatches << " batches of size " << gInferBatchSize << std::endl;
        scoreNetwork(gNbInferBatches, input_num, output_num, inputs, outputs, true /*fp16Mode*/, false /*int8Mode*/);
        gLogInfo << std::endl;
    }

    if (gRunInt8)
    {
        gLogInfo << "INT8 run:" << gNbInferBatches << " batches of size " << gInferBatchSize << std::endl;
        scoreNetwork(gNbInferBatches, input_num, output_num, inputs, outputs, false /*fp16Mode*/, true /*int8Mode*/);
    }

    gLogInfo << "\nComparing FP16 and INT8 Inference accuracy wrt. FP32 Inference accuracy: " << std::endl;
    compareOutputs(outputs);
    gLogInfo << std::endl;

    // shutdown Protobuf
    shutdownProtobufLibrary();

    return gLogger.reportPass(sampleTest);
}
