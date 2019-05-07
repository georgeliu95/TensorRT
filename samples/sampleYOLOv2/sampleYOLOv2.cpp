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
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "common.h"
#include "argsParser.h"
#include "factoryYOLOv2.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using std::vector;

#define CALIBRATION_MODE 0 //Set to '0' for Legacy calibrator and any other value for Entropy calibrator

static const double kCUTOFF = 0.33;   //Cutoff parameter for Legacy calibrator
static const double kQUANTILE = 0.66; //Quantile parameter for Legacy calibrator

const std::string gSampleName = "TensorRT.sample_yolov2";
static samplesCommon::Args gArgs;

// Network variables
static int gImgChannels; // Input image channels
static int gImgHeight;   // Input image height
static int gImgWidth;    // Input image width
static int gGridDim;     // The image is divided into (gGridDim x gGridDim) grid
static int gNumLabel;    // Number of labelled classes in PASCAL VOC
static const char* kINPUT_BLOB_NAME = "data";
static const char* kOUTPUT_BLOB_NAME;

static const char* kNETWORK;
static const char* kBatchPath;
static std::vector<std::string> kDIRECTORIES{"data/yolov2/", "data/samples/yolov2/"}; // Data directory

static const float* kANCHORS;
static const float kANCHORSV2[]{
    0.57273f, 0.677385f, 1.87446f, 2.06253f, 3.33843f, 5.47434f, 7.88282f, 3.52778f, 9.77052f, 9.16828f};
static const float kANCHORS9000[]{
    0.77871f, 1.14074f, 3.00525f, 4.31277f, 9.22725f, 9.61974f};

static std::string* gClassList;
static std::string gClassList_9000[9418];
static std::string gClassList_V2[80]{
    // {{{
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    // }}}
};

// INT8 calibration variables
static const int kCAL_BATCH_SIZE = 1;
static const int kFIRST_CAL_BATCH = 0, kNB_CAL_BATCHES = 100;

// Detections & display
static float gProbThresh;              // Below which the detection will be discarded
static float gNmsThresh;               // NMS threshold
// YOLO9000 paper didn't show a suggested tree threshold value. Set to 0.8 to get a stable label.
// Users can set up values according to their needs.
static const float kTREE_THRESH = 0.8f; // Tree threshold

static bool gCheckYOLO9000 = false;

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

struct PPM
{
    std::string magic, fileName;
    int h, w, max;
    std::vector<uint8_t> buffer;

    PPM(int h, int w, int max, std::string magic)
        : magic(magic)
        , h(h)
        , w(w)
        , max(max)
        , buffer(h * w * 3)
    {
    }

    PPM(const PPM& copy)
        : magic(copy.magic)
        , h(copy.h)
        , w(copy.w)
        , max(copy.max)
        , buffer(copy.buffer)
    {
    }

    ~PPM()
    {
    }
};

struct BBox
{
    float x1, y1, x2, y2;
};

struct BBoxInfo
{
    BBox box;
    int label;
    float prob;
};

std::string locateFile(const std::string& input)
{
    return locateFile(input, kDIRECTORIES);
}

// Simple PPM (portable pixel map) reader
PPM* readPPMFile(const std::string& filename)
{
    std::ifstream infile(locateFile(filename), std::ifstream::binary);
    int h, w, max;
    std::string magic;
    infile >> magic >> w >> h >> max;
    PPM* ppm = new PPM(h, w, max, magic);
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(ppm->buffer.data()), ppm->w * ppm->h * 3);
    return ppm;
}

void drawBox(PPM* ppm, int x1, int x2, int y1, int y2)
{
    for (int x = x1; x <= x2; ++x)
    {
        // Bbox top border
        ppm->buffer[(y1 * ppm->w + x) * 3] = 255;
        ppm->buffer[(y1 * ppm->w + x) * 3 + 1] = 0;
        ppm->buffer[(y1 * ppm->w + x) * 3 + 2] = 0;
        // Bbox bottom border
        ppm->buffer[(y2 * ppm->w + x) * 3] = 255;
        ppm->buffer[(y2 * ppm->w + x) * 3 + 1] = 0;
        ppm->buffer[(y2 * ppm->w + x) * 3 + 2] = 0;
    }
    for (int y = y1; y <= y2; ++y)
    {
        // Bbox left border
        ppm->buffer[(y * ppm->w + x1) * 3] = 255;
        ppm->buffer[(y * ppm->w + x1) * 3 + 1] = 0;
        ppm->buffer[(y * ppm->w + x1) * 3 + 2] = 0;
        // Bbox right border
        ppm->buffer[(y * ppm->w + x2) * 3] = 255;
        ppm->buffer[(y * ppm->w + x2) * 3 + 1] = 0;
        ppm->buffer[(y * ppm->w + x2) * 3 + 2] = 0;
    }
}

//Draw bounding box
void drawBBox(PPM* ppm, const BBox& bbox, int width)
{
    auto round = [](float x) -> int { return int(std::floor(x + 0.5f)); };
    int x1 = std::min(std::max(0, round(int(bbox.x1))), gImgWidth - 1);
    int x2 = std::min(std::max(0, round(int(bbox.x2))), gImgWidth - 1);
    int y1 = std::min(std::max(0, round(int(bbox.y1))), gImgHeight - 1);
    int y2 = std::min(std::max(0, round(int(bbox.y2))), gImgHeight - 1);

    for (int i = 0; i < width; i++)
    {
        drawBox(ppm, x1 + i, x2 - i, y1 + i, y2 - i);
    }
}

void writePPMFile(const std::string& filename, PPM* ppm)
{
    std::ofstream outfile("./" + filename, std::ofstream::binary);
    assert(!outfile.fail());
    outfile << "P6"
            << "\n"
            << ppm->w << " " << ppm->h << "\n"
            << ppm->max << "\n";
    outfile.write(reinterpret_cast<char*>(ppm->buffer.data()), ppm->w * ppm->h * 3);
}

void writePPMFileWithBBox(const std::string& filename, PPM* ppm, const BBox& bbox, int width)
{
    drawBBox(ppm, bbox, width);
    writePPMFile(filename, ppm);
}

int readNamesList(std::string* const classNames, const std::string& filename)
{
    int count = 0;
    std::string name;
    std::ifstream namelist(locateFile(filename));
    if (namelist.is_open())
    {
        while (std::getline(namelist, name))
        {
            classNames[count] = name;
            count++;
        }
    }
    else
        gLogInfo << "Failed to open name list file " << filename << std::endl;
    return count;
}

char* fgetl(FILE* fp)
{
    if (feof(fp))
        return 0;
    size_t size = 512;
    char* line = (char*) malloc(size * sizeof(char));

    if (!fgets(line, size, fp))
    {
        free(line);
        return 0;
    }
    size_t curr = strlen(line);

    while ((line[curr - 1] != '\n') && !feof(fp))
    {
        if (curr == size - 1)
        {
            size *= 2;
            line = (char*) realloc(line, size * sizeof(char));
            if (!line)
            {
                gLogInfo << "Malloc error" << std::endl;
                exit(-1);
            }
        }
        size_t readsize = size - curr;
        if (readsize > INT_MAX)
            readsize = INT_MAX - 1;
        auto val = fgets(&line[curr], readsize, fp);
        assert(val != nullptr);
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n')
        line[curr - 1] = '\0';

    return line;
}

softmaxTree* readTree(const std::string& filename)
{
    softmaxTree t = {0};
    FILE* fp = fopen((locateFile(filename)).c_str(), "r");
    char* line;
    int last_parent = -1;
    int groupSize = 0;
    int groups = 0;
    int n = 0;
    while ((line = fgetl(fp)) != 0)
    {
        char* id = (char*) calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);
        t.parent = (int*) realloc(t.parent, (n + 1) * sizeof(int));
        t.parent[n] = parent;
        t.child = (int*) realloc(t.child, (n + 1) * sizeof(int));
        t.child[n] = -1;
        t.name = (char**) realloc(t.name, (n + 1) * sizeof(char*));
        t.name[n] = id;
        if (parent != last_parent)
        {
            ++groups;
            t.groupOffset = (int*) realloc(t.groupOffset, groups * sizeof(int));
            t.groupOffset[groups - 1] = n - groupSize;
            t.groupSize = (int*) realloc(t.groupSize, groups * sizeof(int));
            t.groupSize[groups - 1] = groupSize;
            groupSize = 0;
            last_parent = parent;
        }
        t.group = (int*) realloc(t.group, (n + 1) * sizeof(int));
        t.group[n] = groups;
        if (parent >= 0)
            t.child[parent] = groups;
        ++n;
        ++groupSize;
    }
    ++groups;
    t.groupOffset = (int*) realloc(t.groupOffset, groups * sizeof(int));
    t.groupOffset[groups - 1] = n - groupSize;
    t.groupSize = (int*) realloc(t.groupSize, groups * sizeof(int));
    t.groupSize[groups - 1] = groupSize;
    t.n = n;
    t.groups = groups;
    t.leaf = (int*) calloc(n, sizeof(int));
    int i;
    for (i = 0; i < n; ++i)
        t.leaf[i] = 1;
    for (i = 0; i < n; ++i)
        if (t.parent[i] >= 0)
            t.leaf[t.parent[i]] = 0;

    fclose(fp);
    softmaxTree* treePtr = (softmaxTree*) calloc(1, sizeof(softmaxTree));
    *treePtr = t;
    return treePtr;
}

void caffeToTRTModel(const std::string& deployFile,                   // name for caffe prototxt
                     const std::string& modelFile,                    // name for model
                     const std::vector<std::string>& outputs,         // network outputs
                     unsigned int maxBatchSize,                       // batch size - NB must be at least as large as the batch we want to run with)
                     nvcaffeparser1::IPluginFactoryV2* pluginFactory, // factory for plugin layers
                     MODE mode,                                       // Precision mode
                     IHostMemory** trtModelStream)                    // output stream for the TensorRT model
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // Parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    INetworkConfig* config = builder->createNetworkConfig();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactoryV2(pluginFactory);
    parser->setProtobufBufferSize(1127_MB);
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
    // Specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(10_MB);
    if (gArgs.runInFp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (gArgs.runInInt8)
    {
        config->setFlag(BuilderFlag::kINT8);
    }
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

#if CALIBRATION_MODE == 0
    CalibrationAlgoType calibrationAlgo = CalibrationAlgoType::kLEGACY_CALIBRATION;
#else
    CalibrationAlgoType calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION_2;
#endif

    if (mode == kINT8)
    {
        BatchStream calibrationStream(kCAL_BATCH_SIZE, kNB_CAL_BATCHES, kBatchPath, kDIRECTORIES);
        
        if (gCheckYOLO9000 && calibrationAlgo == CalibrationAlgoType::kLEGACY_CALIBRATION)
        {
            gLogInfo << "User requested Legacy Calibration with yolo9000. yolo9000 does not support kLEGACY_CALIBRATOR.";
            calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION_2;
        }
        
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
            calibrator.reset(new Int8LegacyCalibrator(calibrationStream, 0, kCUTOFF, kQUANTILE, kNETWORK, true));
        }
        else
        {
            gLogInfo << "Using Entropy Calibrator 2" << std::endl;
            calibrator.reset(new Int8EntropyCalibrator2(calibrationStream, kFIRST_CAL_BATCH, kNETWORK, kINPUT_BLOB_NAME));
        }
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }
    else
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    samplesCommon::enableDLA(builder, config, gArgs.useDLACore);

    gLogInfo << "Begin building engine..." << std::endl;
    auto engine = builder->buildEngineWithConfig(*network, *config);

    assert(engine);
    gLogInfo << "End building engine..." << std::endl;

    // Load tree only after the engine is built. Then the right values will be in place before serialisation.
    if (gCheckYOLO9000)
        gSmTree = readTree("9k.tree");

    // Once the engine is built. Its safe to destroy the calibrator.
    calibrator.reset();

    // We don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down
    (*trtModelStream) = engine->serialize();

    engine->destroy();
    builder->destroy();
    config->destroy();
}

void doInference(IExecutionContext& context, float* inputData, float* predictions, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 1 output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(kINPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(kOUTPUT_BLOB_NAME);

    // Create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * gImgChannels * gImgHeight * gImgWidth * sizeof(float)));                        // Data
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * gGridDim * gGridDim * (gNumBoundBox * (5 + gOutputClsSize)) * sizeof(float))); // Encoded predictions

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));

    auto tStart = std::chrono::high_resolution_clock::now();
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], inputData, batchSize * gImgChannels * gImgHeight * gImgWidth * sizeof(float), cudaMemcpyHostToDevice, stream));
    auto tPreEnqueue = std::chrono::high_resolution_clock::now();
    CHECK(cudaEventRecord(start, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaEventRecord(end, stream));
    auto tPostEnqueue = std::chrono::high_resolution_clock::now();
    CHECK(cudaMemcpyAsync(predictions, buffers[outputIndex], batchSize * gGridDim * gGridDim * (gNumBoundBox * (5 + gOutputClsSize)) * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
    auto tEnd = std::chrono::high_resolution_clock::now();

    float totalGPU{0}, totalEnqueue{0}, totalHost{0};
    CHECK(cudaEventElapsedTime(&totalGPU, start, end));
    totalEnqueue = std::chrono::duration<float, std::milli>(tPostEnqueue - tPreEnqueue).count();
    totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();

    auto oldSettings = gLogInfo.flags();
    auto oldPrecision = gLogInfo.precision();
    gLogInfo << std::fixed << std::setprecision(3) << "GPU time " << totalGPU
             << " ms, host time " << totalHost << " ms, host enqueue time " << totalEnqueue << std::endl;
    gLogInfo.flags(oldSettings);
    gLogInfo.precision(oldPrecision);

    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
}

BBox convertBBox(float i, float j, float fh, float fw,
                 float cx, float cy, float dx, float dy,
                 float bh, float bw)
{
    BBox b;

    float x = (i + cx) / fh;
    float y = (j + cy) / fw;
    float w = exp(dx) * bw / fw;
    float h = exp(dy) * bh / fh;

    b.x1 = x - w / 2u;
    b.x2 = x + w / 2u;

    b.y1 = y - h / 2u;
    b.y2 = y + h / 2u;

    return b;
}

size_t getFeatureIndex(int b, int f, int h, int w,
                       int F, int H, int W)
{
    return (b * F * H * W) + (f * H * W) + (h * W) + w;
}

void hierarchyPredictions(float* predictions, int n, softmaxTree* hier, int only_leaves, int stride)
{
    int j;
    for (j = 0; j < n; ++j)
    {
        int parent = hier->parent[j];
        if (parent >= 0)
            predictions[j * stride] *= predictions[parent * stride];
    }
    if (only_leaves)
    {
        for (j = 0; j < n; ++j)
            if (!hier->leaf[j])
                predictions[j * stride] = 0;
    }
}

int hierarchyTopPrediction(float* predictions, softmaxTree* hier, float thresh, int stride)
{
    int group = 0;
    while (1)
    {
        float max = 0.;
        int max_i = 0;
        for (int i = 0; i < hier->groupSize[group]; ++i)
        {
            int index = i + hier->groupOffset[group];
            float val = predictions[(i + hier->groupOffset[group]) * stride];
            if (val > max)
            {
                max_i = index;
                max = val;
            }
        }
        if (max > thresh)
        {
            group = hier->child[max_i];
            if (hier->child[max_i] < 0)
                return max_i;
        }
        else
            return hier->parent[hier->groupOffset[group]];
    }
    return 0;
}

std::vector<BBoxInfo> decodeDetections(const int S, const int B, const int CLASSES, const int inputH, const int inputW,
                                       const float probThresh, const float treeThresh, softmaxTree* smTree, float* detections)
{
    std::vector<BBoxInfo> binfo;
    for (int y = 0; y < S; y++)
    {
        for (int x = 0; x < S; x++)
        {
            for (int b = 0; b < B; b++)
            {
                float cx, cy, dx, dy;
                cx = detections[getFeatureIndex(b, 0, y, x, (4 + 1 + CLASSES), S, S)];
                cy = detections[getFeatureIndex(b, 1, y, x, (4 + 1 + CLASSES), S, S)];
                dx = detections[getFeatureIndex(b, 2, y, x, (4 + 1 + CLASSES), S, S)];
                dy = detections[getFeatureIndex(b, 3, y, x, (4 + 1 + CLASSES), S, S)];

                float objectness = detections[getFeatureIndex(b, 4, y, x, (4 + 1 + CLASSES), S, S)];

                float maxProb{};
                int maxIndex{};

                // YOLOv2 branch
                if (smTree == nullptr)
                {
                    for (int i = 0; i < CLASSES; i++)
                    {
                        float prob = detections[getFeatureIndex(b, 5 + i, y, x, (4 + 1 + CLASSES), S, S)];

                        if (prob > maxProb)
                        {
                            maxProb = prob;
                            maxIndex = i;
                        }
                    }
                    maxProb = objectness * maxProb;
                }
                // YOLO9000 branch
                else
                {
                    hierarchyPredictions(detections + getFeatureIndex(b, 5, y, x, (4 + 1 + CLASSES), S, S), CLASSES, smTree, 0, S * S);
                    maxIndex = hierarchyTopPrediction(detections + getFeatureIndex(b, 5, y, x, (4 + 1 + CLASSES), S, S), smTree, treeThresh, S * S);
                    maxProb = (objectness > probThresh) ? objectness : 0.;
                }

                if (maxProb > probThresh)
                {
                    BBoxInfo bbi;
                    bbi.box = convertBBox(x, y, S, S,
                                          cx, cy, dx, dy,
                                          kANCHORS[2 * b + 1], kANCHORS[2 * b]);
                    bbi.box.x1 *= inputW;
                    bbi.box.x2 *= inputW;
                    bbi.box.y1 *= inputH;
                    bbi.box.y2 *= inputH;
                    bbi.label = maxIndex;
                    bbi.prob = maxProb;

                    binfo.push_back(bbi);
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

    std::stable_sort(binfo.begin(), binfo.end(), [](const BBoxInfo& b1, const BBoxInfo& b2) {
        return b1.prob > b2.prob;
    });
    vector<BBoxInfo> out;
    for (auto i : binfo)
    {
        bool keep = std::all_of(out.begin(), out.end(),
                                [&](BBoxInfo& j) { return computeIoU(i.box, j.box) <= nmsThresh; });
        if (keep)
            out.push_back(i);
    }
    return out;
}

int findArg(int argc, char* argv[], const char* arg)
{
    for (int i = 0; i < argc; ++i)
    {
        if (!argv[i])
            continue;
        if (0 == strcmp(argv[i], arg))
            return 1;
    }
    return 0;
}

bool prepareYOLO9000()
{
    gImgChannels = 3;
    gImgHeight = 544;
    gImgWidth = 544;
    gOutputClsSize = 9418;
    gGridDim = 17;
    gNumBoundBox = 3;
    gNumLabel = gNumBoundBox * (4 + 1 + gOutputClsSize);

    kANCHORS = kANCHORS9000;
    kOUTPUT_BLOB_NAME = (char*) "region20";
    kNETWORK = (char*) "yolo9000";
    kBatchPath = (char*) "./batches9000/batch_calibration";

    if (readNamesList(gClassList_9000, "9k.names") != gOutputClsSize)
    {
        gLogError << "names list size expected to be " << gOutputClsSize << std::endl;
        return false;
    }

    gClassList = gClassList_9000;  // Init gClassList by reading from names list file
    gSmTree = readTree("9k.tree"); // Init softmax tree for region layer

    gProbThresh = 0.24f;
    gNmsThresh = 0.4f;

    return true;
}

bool prepareYOLOv2()
{
    gImgChannels = 3;
    gImgHeight = 416;
    gImgWidth = 416;
    gOutputClsSize = 80;
    gGridDim = 13;
    gNumBoundBox = 5;
    gNumLabel = gNumBoundBox * (4 + 1 + gOutputClsSize);

    kANCHORS = kANCHORSV2;
    kOUTPUT_BLOB_NAME = (char*) "region26";
    kNETWORK = (char*) "yolov2";
    kBatchPath = (char*) "./batchesV2/batch_calibration";
    gClassList = gClassList_V2; // Init gClassList
    gSmTree = nullptr;          // Init softmax tree for region layer

    gProbThresh = 0.2f;
    gNmsThresh = 0.4f;

    return true;
}

void parseOptions(int argc, char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        char* optName = argv[i];
        std::string arg(argv[i]);
        if (0 == strcmp(optName, "--yolo9000"))
        {
            gCheckYOLO9000 = true;
        }
    }
}

void printHelp(const char* name)
{
    std::cout << "Usage: ./" << name << "\n"
        << "  -h --help         Display help information.\n"
        << "  --useDLACore=N    Specify the DLA engine to run on.\n"
        << "  --fp16            Specify to run in fp16 mode.\n"
        << "  --int8            Specify to run in int8 mode.\n"
        << "  --yolo9000        Run yolo9000 instead of yolov2.\n";
}
int main(int argc, char** argv)
{
    parseOptions(argc, argv);
    std::vector<char*> argVector;
    for (int x = 0; x < argc; ++x)
    {
        if (std::string(argv[x]) != "--yolo9000")
        {
            argVector.push_back(argv[x]);
        }
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);
    bool argsOK = samplesCommon::parseArgs(gArgs, argVector.size(), argVector.data());
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


    gLogger.reportTestStart(sampleTest);

    if (gCheckYOLO9000)
    {
        gLogInfo << "Sample YOLO9000" << std::endl;
        kDIRECTORIES.push_back("data/int8_samples/yolo9000/");
        kDIRECTORIES.push_back("int8/yolo9000/");
        if (!prepareYOLO9000())
            return gLogger.reportFail(sampleTest);
    }
    else // YOLOv2
    {
        gLogInfo << "Sample YOLOv2" << std::endl;
        kDIRECTORIES.push_back("data/int8_samples/yolov2/");
        kDIRECTORIES.push_back("int8/yolov2/");
        if (!prepareYOLOv2())
            return gLogger.reportFail(sampleTest);
    }

    const int N = 1; // batch size
    // Create a TensorRT model from the caffe model and serialize it to a stream
    YOLOv2PluginFactory pluginFactorySerialize;
    IHostMemory* trtModelStream{nullptr};
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    caffeToTRTModel(gCheckYOLO9000 ? "yolo9000_trt.prototxt" : "yolo_trt.prototxt",
                    gCheckYOLO9000 ? "yolo9000.caffemodel" : "yolo.caffemodel",
                    std::vector<std::string>{kOUTPUT_BLOB_NAME},
                    N, &pluginFactorySerialize, params.modelType, &trtModelStream);

    assert(trtModelStream != nullptr);
    pluginFactorySerialize.destroyPlugin();

    // Deserialize the engine
    YOLOv2PluginFactory pluginFactory;
    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Available images
    std::vector<std::string> imageList;
    if (gCheckYOLO9000)
        imageList = {"dog9000.ppm"}; // Input image list for YOLO9000. Each image size is 544 * 544
    else
        imageList = {"dog.ppm"}; // // Input image list for YOLOv2. Each image size is 416 * 416
    std::vector<PPM*> ppms;

    for (int i = 0; i < N; ++i)
    {
        ppms.push_back(readPPMFile(imageList[i]));
    }

    std::vector<float> data(N * gImgChannels * gImgHeight * gImgWidth);
    for (int i = 0, volImg = gImgChannels * gImgHeight * gImgWidth; i < N; ++i)
    {
        for (int c = 0; c < gImgChannels; ++c)
        {
            // The color image to input should be in RGB order
            for (unsigned j = 0, volChl = gImgHeight * gImgWidth; j < volChl; ++j)
            {
                data[i * volImg + c * volChl + j] = ppms[i]->buffer[j * gImgChannels + c] / 255.;
            }
        }
    }

    // Host memory for outputs
    const int outputDim = gGridDim * gGridDim * (gNumBoundBox * (4 + 1 + gOutputClsSize));
    std::vector<float> detections(N * outputDim);

    // Run inference
    doInference(*context, data.data(), detections.data(), N);

    bool pass = true;
    for (int p = 0; p < N; ++p)
    {
        auto binfo = decodeDetections(gGridDim, gNumBoundBox, gOutputClsSize, gImgHeight, gImgWidth,
                                      gProbThresh, kTREE_THRESH, gSmTree, &detections[p * outputDim]);

        auto remaining = nms(gNmsThresh, binfo);
        pass &= remaining.size() >= 1;
        // is there at least one correct detection?
        bool correctDetection = false;

        for (auto b : remaining)
        {
            if (gClassList[b.label] == "dog" || gClassList[b.label] == "carnivore")
                correctDetection = true;
            gLogInfo << " Image name: " << imageList[p] << ", Label: " << gClassList[b.label] << ","
                     << " confidence: " << b.prob
                     << " xmin: " << (b.box.x1)
                     << " ymin: " << (b.box.y1)
                     << " xmax: " << (b.box.x2)
                     << " ymax: " << (b.box.y2)
                     << std::endl;
            std::string storeName = imageList[p] + "-" + gClassList[b.label] + "-" + std::to_string(b.prob) + "_" + imageList[p];
            PPM* cpy = new PPM(*(ppms[p]));
            writePPMFileWithBBox(storeName, cpy, b.box, 3); // Create output image with bounding boxes
            gLogInfo << "Result stored in " << storeName << "." << std::endl;
            delete cpy;
        }
        pass &= correctDetection;
    }

    for (auto ppm : ppms)
        delete ppm;
    ppms.clear();

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Note: Once you call shutdownProtobufLibrary, you cannot use the parsers anymore.
    shutdownProtobufLibrary();

    return gLogger.reportTest(sampleTest, pass);
}
