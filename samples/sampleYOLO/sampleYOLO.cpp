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
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>

#include "BatchStream.h"
#include "Calibrator.h"
#include "logger.h"
#include "common.h"
#include "factoryYOLO.h"
#include "argsParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace std;

using namespace samplesCommon;
static const double kCUTOFF = 0.21;
static const double kQUANTILE = 0.9999;

#define CALIBRATION_MODE 0 //Set to '0' for Legacy calibrator and any other value for Entropy calibrator

const std::string gSampleName = "TensorRT.sample_yolo";

static samplesCommon::Args gArgs;

// YOLO network variables
static const int kINPUT_C = 3;                      // input image channels
static const int kINPUT_H = 448;                    //input image height
static const int kINPUT_W = 448;                    //input image width
static const int kOUTPUT_CLS_SIZE = 21;             // number of classes
static const int kGRID_DIM = 7;                     // the image is divided into (kGRID_DIM x kGRID_DIM) grid
static const int kBOUNDING_BOX = 2;                 // each grid cell predicts kBOUNDING_BOX bounding boxes
static const int kNUM_LABEL = kOUTPUT_CLS_SIZE - 1; // number of labelled classes in PASCAL VOC
const char* gNetworkName = "yolo";                  //network name

const char* const gCLASSES[kOUTPUT_CLS_SIZE]{"background", "aeroplane", "bicycle", "bird", "boat",
                                             "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                             "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                             "sofa", "train", "tvmonitor"};             // list of class labels
static const char* kINPUT_BLOB_NAME = "data";                                           // input blob name
static const char* kOUTPUT_BLOB_NAME = "detections";                                    //output blob name
static const std::vector<std::string> kDIRECTORIES{"data/yolo/", "data/samples/yolo/", "data/int8_samples/yolo/", "int8/yolo/"}; // data directory

// INT8 calibration variables
static const int kCAL_BATCH_SIZE = 1;
static const int kFIRST_CAL_BATCH = 0, kNB_CAL_BATCHES = 100;

// detections & display
static const float kPROB_THRESH = 0.2f; // below which the detection will be discarded
static const float kNMS_THRESH = 0.4f;  // NMS threshold

struct BBoxInfo
{
    BBox box;
    int label;
    float prob;
};

enum MODE
{
    kFP32,
    kFP16,
    kINT8,
    kUNKNOWN
};

struct Param
{
    MODE modelType{MODE::kFP32}; //Default run FP32 precision
} params;

std::ostream& operator<<(std::ostream& o, MODE dt)
{
    switch (dt)
    {
    case kFP32: o << "FP32"; break;
    case kFP16: o << "FP16"; break;
    case kINT8: o << "INT8"; break;
    case kUNKNOWN: o << "UNKNOWN"; break;
    }
    return o;
}

std::string locateFile(const std::string& input)
{
    return locateFile(input, kDIRECTORIES);
}

void caffeToTRTModel(const std::string& deployFile,                   // name for caffe prototxt
                     const std::string& modelFile,                    // name for model
                     const std::vector<std::string>& outputs,         // network outputs
                     unsigned int maxBatchSize,                       // batch size - NB must be at least as large as the batch we want to run with)
                     nvcaffeparser1::IPluginFactoryV2* pluginFactory, // factory for plugin layers
                     MODE mode,                                       // Precision mode
                     IHostMemory** trtModelStream)                    // output stream for the TensorRT model
{
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    INetworkConfig* config = builder->createNetworkConfig();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactoryV2(pluginFactory);
    parser->setProtobufBufferSize(1127_MB); // 1.1GB

    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(10_MB);
    DataType dataType = DataType::kFLOAT;
    if (mode == kFP16)
        dataType = DataType::kHALF;
    gLogInfo << "Begin parsing model..." << std::endl;
    gLogInfo << mode << " mode running..." << std::endl;

    const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
                                                              locateFile(modelFile).c_str(),
                                                              *network,
                                                              dataType);
    gLogInfo << "End parsing model..." << std::endl;
    // specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    
    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

#if CALIBRATION_MODE == 0
    CalibrationAlgoType calibrationAlgo = CalibrationAlgoType::kLEGACY_CALIBRATION;
#else
    CalibrationAlgoType calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION_2;
#endif

    if (mode == kINT8)
    {
        BatchStream calibrationStream(kCAL_BATCH_SIZE, kNB_CAL_BATCHES, "./batches/batch_calibration", kDIRECTORIES);
        
        if (gArgs.useDLACore >= 0)
        {        
            if (calibrationAlgo == CalibrationAlgoType::kLEGACY_CALIBRATION)
            {
                gLogInfo << "User requested Legacy Calibration with DLA. DLA only supports kENTROPY_CALIBRATOR_2.";
            }
            gLogInfo << " DLA requested. Setting Calibrator to use kENTROPY_CALIBRATOR_2." << std::endl;
            calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION_2;
        }
        
        if (calibrationAlgo == CalibrationAlgoType::kLEGACY_CALIBRATION)
        {
            gLogInfo << "Using Legacy Calibrator" << std::endl;
            calibrator.reset(new Int8LegacyCalibrator(calibrationStream, 0, kCUTOFF, kQUANTILE, gNetworkName, true));
        }
        else
        {
            gLogInfo << "Using Entropy Calibrator 2" << std::endl;
            calibrator.reset(new Int8EntropyCalibrator2(calibrationStream, kFIRST_CAL_BATCH, gNetworkName, kINPUT_BLOB_NAME));
        }
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }
    else
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    samplesCommon::enableDLA(builder, config, gArgs.useDLACore);

    gLogInfo << "Begin building engine..." << std::endl;
    auto engine = builder->buildEngineWithConfig(*network, *config);

    assert(engine);
    gLogInfo << "End building engine..." << std::endl;

    // Once the engine is built. Its safe to destroy the calibrator.
    calibrator.reset();

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    (*trtModelStream) = engine->serialize();

    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* inputData, float* predictions, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 1 output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(kINPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(kOUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * kINPUT_C * kINPUT_H * kINPUT_W * sizeof(float)));                            // input data
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * kGRID_DIM * kGRID_DIM * (kBOUNDING_BOX * 5 + kNUM_LABEL) * sizeof(float))); // encoded predictions
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * kINPUT_C * kINPUT_H * kINPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(predictions, buffers[outputIndex], batchSize * kGRID_DIM * kGRID_DIM * (kBOUNDING_BOX * 5 + kNUM_LABEL) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

std::vector<BBoxInfo> decodeDetections(const int S, const int B, const int C, const int inputH, const int inputW, const float probThresh, const float* detections)
{
    // detections' layout: SxSxC (classes) + SxSxB (probs for B BBoxes) + SxSxBx4 (coordinates for B BBoxes)
    std::vector<float> probObj(S * S * B);
    std::vector<int> labels(S * S);
    for (int i = 0; i < S; ++i)
    {
        for (int j = 0; j < S; ++j)
        {
            const float* det = &detections[(i * S + j) * C];
            auto iter = std::max_element(det, det + C);
            labels[i * S + j] = std::distance(det, iter) + 1;
            probObj[(i * S + j) * B] = *iter * detections[S * S * C + (i * S + j) * B];
            probObj[(i * S + j) * B + 1] = *iter * detections[S * S * C + (i * S + j) * B + 1];
        }
    }
    std::vector<BBoxInfo> binfo;
    for (int i = 0; i < S; ++i)
    {
        for (int j = 0; j < S; ++j)
        {
            for (int k = 0; k < B; ++k)
            {
                if (probObj[(i * S + j) * B + k] > probThresh)
                {
                    const float* det = &detections[S * S * C + S * S * B + ((i * S + j) * B + k) * 4];
                    float ctrX, ctrY, width, height;
                    BBoxInfo c;
                    ctrX = inputW * (det[0] + j) / S;
                    ctrY = inputH * (det[1] + i) / S;
                    width = inputW * det[2] * det[2];
                    height = inputH * det[3] * det[3];
                    c.box.x1 = ctrX - width / 2;
                    c.box.y1 = ctrY - height / 2;
                    c.box.x2 = ctrX + width / 2;
                    c.box.y2 = ctrY + height / 2;
                    c.label = labels[i * S + j];
                    c.prob = probObj[(i * S + j) * B + k];
                    binfo.push_back(c);
                }
            }
        }
    }
    return binfo;
}

std::vector<BBoxInfo> nms(const float nmsThresh, std::vector<BBoxInfo> binfo)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto computeIoU = [&overlap1D](BBox& bbox1, BBox& bbox2) -> float {
        float overlapX = overlap1D(bbox1.x1, bbox1.x2, bbox2.x1, bbox2.x2);
        float overlapY = overlap1D(bbox1.y1, bbox1.y2, bbox2.y1, bbox2.y2);
        float area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
        float area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    vector<BBoxInfo> out;
    std::map<int, std::vector<BBoxInfo>> binfo_tot_class;
    for (auto i : binfo)
    {
        binfo_tot_class[i.label].push_back(i);
    }
    for (auto t : binfo_tot_class)
    {
        std::vector<BBoxInfo> binfo_each_class;
        std::vector<BBoxInfo>& v = t.second;
        std::stable_sort(v.begin(), v.end(), [](const BBoxInfo& b1, const BBoxInfo& b2) {
            return b1.prob > b2.prob;
        });

        for (auto i : v)
        {
            bool keep = std::all_of(binfo_each_class.begin(), binfo_each_class.end(), [&](BBoxInfo& j) { return computeIoU(i.box, j.box) <= nmsThresh; });
            if (keep)
                binfo_each_class.push_back(i);
        }

        out.insert(out.end(), binfo_each_class.begin(), binfo_each_class.end());
    }
    return out;
}

void printHelp(const char* name)
{
    std::cout << "Usage: ./" << name << "\n"
        << "  -h --help         Display help information.\n"
        << "  --useDLACore=N    Specify the DLA engine to run on.\n"
        << "  --fp16            Specify to run in fp16 mode.\n"
        << "  --int8            Specify to run in int8 mode.\n";
}


int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help || !argsOK)
    {
        printHelp(argv[0]);
        return argsOK ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    params.modelType = kFP32;
    if (gArgs.runInFp16)
    {
        params.modelType = kFP16;
    }
    else if (gArgs.runInInt8)
    {
        params.modelType = kINT8;
    }



    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    const int N = 1;                                     // select batch size
    std::vector<std::string> imageList = {"person.ppm"}; // input ImageList with image size 448 * 448
    std::vector<PPM<kINPUT_C, kINPUT_H, kINPUT_W>> ppms(N);
    YOLOPluginFactory pluginFactorySerialize;
    IHostMemory* trtModelStream{nullptr};
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    // Create a TensorRT model from the caffe model and serialize it to a stream
    caffeToTRTModel("yolo.prototxt",
                    "yolo.caffemodel",
                    std::vector<std::string>{kOUTPUT_BLOB_NAME},
                    N, &pluginFactorySerialize, params.modelType, &trtModelStream);

    assert(trtModelStream != nullptr);
    pluginFactorySerialize.destroyPlugin();

    gLogInfo << "Deserialization started" << std::endl;
    // Deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    gLogInfo << "Deserialization end" << std::endl;
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    //Read 'N' image files from 'imageList' vector
    for (int i = 0; i < N; ++i)
    {
        readPPMFile(locateFile(imageList[i]), ppms[i]);
    }

    std::vector<float> data(N * kINPUT_C * kINPUT_H * kINPUT_W); // Input Image buffer
    for (int i = 0, volImg = kINPUT_C * kINPUT_H * kINPUT_W; i < N; ++i)
    {
        for (int c = 0; c < kINPUT_C; ++c)
        {
            // The color image to input should be in RGB order
            for (unsigned j = 0, volChl = kINPUT_H * kINPUT_W; j < volChl; ++j)
            {
                data[i * volImg + c * volChl + j] = ppms[i].buffer[j * kINPUT_C + c] / 127.5f - 1;
            }
        }
    }

    // Host memory for outputs
    const int outputDim = kGRID_DIM * kGRID_DIM * (kBOUNDING_BOX * 5 + kNUM_LABEL);
    std::vector<float> detections(N * outputDim);

    // Run inference
    doInference(*context, data.data(), detections.data(), N);

    bool pass = true;

    for (int p = 0; p < N; ++p)
    {
        auto binfo = decodeDetections(kGRID_DIM, kBOUNDING_BOX, kNUM_LABEL, kINPUT_H, kINPUT_W, kPROB_THRESH, &detections[p * outputDim]);
        auto remaining = nms(kNMS_THRESH, binfo);
        pass &= remaining.size() >= 1;
        // is there at least one correct detection?
        bool correctDetection = false;
        for (auto b : remaining)
        {
            if (std::string(gCLASSES[b.label]) == "person")
                correctDetection = true;
            gLogInfo << " Image name: " << imageList[p] << ", Label: " << gCLASSES[b.label] << ","
                     << " confidence: " << b.prob
                     << " xmin: " << (b.box.x1)
                     << " ymin: " << (b.box.y1)
                     << " xmax: " << (b.box.x2)
                     << " ymax: " << (b.box.y2)
                     << std::endl;
            std::string storeName = imageList[p] + "-" + gCLASSES[b.label] + "-" + std::to_string(b.prob) + ".ppm";
            writePPMFileWithBBox(storeName, ppms[p], b.box);
            gLogInfo << "Result stored in " << storeName << "." << std::endl;
        }
        pass &= correctDetection;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Note: Once you call shutdownProtobufLibrary, you cannot use the parsers anymore.
    shutdownProtobufLibrary();

    return gLogger.reportTest(sampleTest, pass);
}
