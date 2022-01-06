/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! samplePointPillars.cpp
//! This file contains the implementation of the ONNX PointPillars sample. It creates the network using
//! the PointPillar ONNX model.
//! It can be run with the following command line:
//! Command: ./sample_point_pillars [-h]
//!

#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "argsParser.h"
#include "common.h"
#include "parserOnnxConfig.h"
#include "buffers.h"

using namespace samplesCommon;
using samplesCommon::SampleUniquePtr;

//! \class
//!
//! \brief Define the 3D bounding box.
//!
struct Bndbox3D {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    Bndbox3D(){};
    Bndbox3D(float x_, float y_, float z_, float l_, float w_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};

//! \brief The name of this sample.
//!
const std::string gSampleName = "TensorRT.sample_point_pillars";

//! \class
//!
//! \brief Define the parameters for this sample.
//!
struct SamplePointPillarsParams : public samplesCommon::SampleParams
{
  std::string modelFile;
  std::string dataFile;
  std::vector<std::string> classNames;
  int maxPointsNum;
  int numPointFeatures;
  float nmsIoUThreshold;
  int preNMSTopN;
  bool doProfile;
  std::string saveEngine;
  std::string loadEngine;
};

//! \class
//!
//! \brief The class that defines the overall workflow of this sample.
//!
class SamplePointPillars
{
public:
    SamplePointPillars(const SamplePointPillarsParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    SamplePointPillarsParams mParams; //!< The parameters for the sample.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    const float ThresHold = 1e-8; //!< The threshold for post processing.

    //!
    //! \brief Parses an ONNX model and creates a TensorRT network
    //!
    bool constructNetwork(
        SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
        SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Filters output detections and verify results
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Read point cloud points from file
    //!
    int loadData(const char *file, char* data);
    //!
    //! \brief Compute cross product of two vectors
    //!
    float cross(const float2 p1, const float2 p2, const float2 p0);

    //!
    //! \brief Check if a point is in a box
    //!
    int check_box2d(const Bndbox3D box, const float2 p);

    //!
    //! \brief Intersection of lines
    //!
    bool intersection(const float2 p1, const float2 p0, const float2 q1,
                      const float2 q0, float2 &ans);

    //!
    //! \brief Rotate box around center
    //!
    void rotate_around_center(const float2 &center, const float angle_cos,
                              const float angle_sin, float2 &p);

    //!
    //! \brief Box overlap
    //!
    float box_overlap(const Bndbox3D &box_a, const Bndbox3D &box_b);

    //!
    //! \brief 3D NMS on CPU
    //!
    int nms_cpu(std::vector<Bndbox3D> &bndboxes, const float nms_thresh,
                std::vector<Bndbox3D> &nms_pred, const int pre_nms_top_n);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the ONNX PointPillars network by parsing the ONNX model and builds
//!          the engine that will be used to run PointPillars(mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SamplePointPillars::build()
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    if (!mParams.loadEngine.empty())
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file(mParams.loadEngine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }
        IRuntime* infer = nvinfer1::createInferRuntime(sample::gLogger);
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            infer->deserializeCudaEngine(
                trtModelStream.data(), size, nullptr
            ),
            samplesCommon::InferDeleter()
        );
        infer->destroy();
        sample::gLogInfo << "TRT Engine loaded from: " << mParams.loadEngine << std::endl;
        if (!mEngine)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }
    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }
    ASSERT(network->getNbInputs() == 2);
    auto dims0 = network->getInput(0)->getDimensions();
    ASSERT(dims0.nbDims == 3);
    auto dims1 = network->getInput(1)->getDimensions();
    ASSERT(dims1.nbDims == 1);

    ASSERT(network->getNbOutputs() == 2);
    auto out_dims0 = network->getOutput(0)->getDimensions();
    ASSERT(out_dims0.nbDims == 3);
    auto out_dims1 = network->getOutput(1)->getDimensions();
    ASSERT(out_dims1.nbDims == 1);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the ONNX Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the ONNX network
//!
//! \param builder Pointer to the engine builder
//!
bool SamplePointPillars::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.modelFile, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    nvinfer1::Dims dims{};
    dims.nbDims = 3;
    dims.d[0] = mParams.batchSize;
    dims.d[1] = mParams.maxPointsNum;
    dims.d[2] = mParams.numPointFeatures;
    profile->setDimensions(mParams.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions(mParams.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions(mParams.inputTensorNames[0].c_str(), nvinfer1::OptProfileSelector::kMAX, dims);
    dims.nbDims = 1;
    dims.d[0] = mParams.batchSize;
    profile->setDimensions(mParams.inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMIN, dims);
    profile->setDimensions(mParams.inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kOPT, dims);
    profile->setDimensions(mParams.inputTensorNames[1].c_str(), nvinfer1::OptProfileSelector::kMAX, dims);
    config->addOptimizationProfile(profile);

    config->setMaxWorkspaceSize(2_GiB);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);
    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }
    if (!mParams.saveEngine.empty())
    {
        std::ofstream p(mParams.saveEngine, std::ios::binary);
        if (!p)
        {
            return false;
        }
        nvinfer1::IHostMemory* ptr = mEngine->serialize();
        ASSERT(ptr);
        p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
        ptr->destroy();
        p.close();
        sample::gLogInfo << "TRT Engine file saved to: " << mParams.saveEngine << std::endl;
    }
    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SamplePointPillars::infer()
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }
    SimpleProfiler profiler("PointPillars performance");
    if (mParams.doProfile)
    {
        context->setProfiler(&profiler);
    }
    // Read the input data into the managed buffers
    if (!processInput(buffers))
    {
        return false;
    }
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }
    if (mParams.doProfile)
        std::cout << profiler;
    return true;
}

int SamplePointPillars::loadData(const char *file, char* data)
{
  std::fstream dataFile(file, std::ifstream::in);
  if (!dataFile.is_open())
  {
    return -1;
  }
  int len = 0;
  dataFile.seekg(0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg(0, dataFile.beg);
  dataFile.read(data, len);
  dataFile.close();
  return len;
}

bool SamplePointPillars::processInput(const samplesCommon::BufferManager& buffers)
{
    float* pointsBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int32_t* numPointsBuffer = static_cast<int32_t*>(buffers.getHostBuffer(mParams.inputTensorNames[1]));
    numPointsBuffer[0] = loadData(locateFile(mParams.dataFile, mParams.dataDirs).c_str(), (char*)pointsBuffer)
                         / sizeof(float) / mParams.numPointFeatures;
    return true;
}

float SamplePointPillars::cross(const float2 p1, const float2 p2, const float2 p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

int SamplePointPillars::check_box2d(const Bndbox3D box, const float2 p) {
    const float MARGIN = 1e-2;
    float center_x = box.x;
    float center_y = box.y;
    float angle_cos = cos(-box.rt);
    float angle_sin = sin(-box.rt);
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;
    return (fabs(rot_x) < box.l / 2 + MARGIN && fabs(rot_y) < box.w / 2 + MARGIN);
}

bool SamplePointPillars::intersection(const float2 p1, const float2 p0, const float2 q1, const float2 q0, float2 &ans) {
    if (( std::min(p0.x, p1.x) <= std::max(q0.x, q1.x) &&
          std::min(q0.x, q1.x) <= std::max(p0.x, p1.x) &&
          std::min(p0.y, p1.y) <= std::max(q0.y, q1.y) &&
          std::min(q0.y, q1.y) <= std::max(p0.y, p1.y) ) == 0)
        return false;
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);
    if (!(s1 * s2 > 0 && s3 * s4 > 0))
        return false;
    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > ThresHold) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);
    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;
        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }
    return true;
}

void SamplePointPillars::rotate_around_center(const float2 &center, const float angle_cos, const float angle_sin, float2 &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p = float2 {new_x, new_y};
}

float SamplePointPillars::box_overlap(const Bndbox3D &box_a, const Bndbox3D &box_b) {
    float a_angle = box_a.rt, b_angle = box_b.rt;
    float a_dx_half = box_a.l / 2, b_dx_half = box_b.l / 2, a_dy_half = box_a.w / 2, b_dy_half = box_b.w / 2;
    float a_x1 = box_a.x - a_dx_half, a_y1 = box_a.y - a_dy_half;
    float a_x2 = box_a.x + a_dx_half, a_y2 = box_a.y + a_dy_half;
    float b_x1 = box_b.x - b_dx_half, b_y1 = box_b.y - b_dy_half;
    float b_x2 = box_b.x + b_dx_half, b_y2 = box_b.y + b_dy_half;
    float2 box_a_corners[5];
    float2 box_b_corners[5];
    float2 center_a = float2 {box_a.x, box_a.y};
    float2 center_b = float2 {box_b.x, box_b.y};
    float2 cross_points[16];
    float2 poly_center =  {0, 0};
    int cnt = 0;
    bool flag = false;
    box_a_corners[0] = float2 {a_x1, a_y1};
    box_a_corners[1] = float2 {a_x2, a_y1};
    box_a_corners[2] = float2 {a_x2, a_y2};
    box_a_corners[3] = float2 {a_x1, a_y2};
    box_b_corners[0] = float2 {b_x1, b_y1};
    box_b_corners[1] = float2 {b_x2, b_y1};
    box_b_corners[2] = float2 {b_x2, b_y2};
    box_b_corners[3] = float2 {b_x1, b_y2};
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);
    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }
    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                                box_b_corners[j + 1], box_b_corners[j],
                                cross_points[cnt]);
            if (flag) {
                poly_center = {poly_center.x + cross_points[cnt].x, poly_center.y + cross_points[cnt].y};
                cnt++;
            }
        }
    }
    for (int k = 0; k < 4; k++) {
        if (check_box2d(box_a, box_b_corners[k])) {
            poly_center = {poly_center.x + box_b_corners[k].x, poly_center.y + box_b_corners[k].y};
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_box2d(box_b, box_a_corners[k])) {
            poly_center = {poly_center.x + box_a_corners[k].x, poly_center.y + box_a_corners[k].y};
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }
    poly_center.x /= cnt;
    poly_center.y /= cnt;
    float2 temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (atan2(cross_points[i].y - poly_center.y, cross_points[i].x - poly_center.x) >
                atan2(cross_points[i+1].y - poly_center.y, cross_points[i+1].x - poly_center.x)
                ) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }
    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        float2 a = {cross_points[k].x - cross_points[0].x,
                    cross_points[k].y - cross_points[0].y};
        float2 b = {cross_points[k + 1].x - cross_points[0].x,
                    cross_points[k + 1].y - cross_points[0].y};
        area += (a.x * b.y - a.y * b.x);
    }
    return fabs(area) / 2.0;
}

int SamplePointPillars::nms_cpu(std::vector<Bndbox3D> &bndboxes, const float nms_thresh,
                                std::vector<Bndbox3D> &nms_pred, const int pre_nms_top_n)
{
    std::sort(bndboxes.begin(), bndboxes.end(),
              [](Bndbox3D boxes1, Bndbox3D boxes2) { return boxes1.score > boxes2.score; });
    std::vector<int> suppressed(std::min(int(bndboxes.size()), pre_nms_top_n), 0);
    for (size_t i = 0; i < std::min(int(bndboxes.size()), pre_nms_top_n); i++) {
        if (suppressed[i] == 1) {
            continue;
        }
        nms_pred.emplace_back(bndboxes[i]);
        for (size_t j = i + 1; j < std::min(int(bndboxes.size()), pre_nms_top_n); j++) {
            if (suppressed[j] == 1) {
                continue;
            }
            float sa = bndboxes[i].l * bndboxes[i].w;
            float sb = bndboxes[j].l * bndboxes[j].w;
            float s_overlap = box_overlap(bndboxes[i], bndboxes[j]);
            float iou = s_overlap / fmaxf(sa + sb - s_overlap, ThresHold);
            if (iou >= nms_thresh) {
                suppressed[j] = 1;
            }
        }
    }
    return 0;
}

bool SamplePointPillars::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const float* boxes = static_cast<const float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    const int32_t* num_boxes = static_cast<const int32_t*>(buffers.getHostBuffer(mParams.outputTensorNames[1]));
    std::vector<Bndbox3D> res;
    std::vector<Bndbox3D> postNMSBoxes;
    for (int i = 0; i < num_boxes[0]; i++) {
        auto Bb = Bndbox3D(
            boxes[i * 9],
            boxes[i * 9 + 1],
            boxes[i * 9 + 2],
            boxes[i * 9 + 3],
            boxes[i * 9 + 4],
            boxes[i * 9 + 5],
            boxes[i * 9 + 6],
            boxes[i * 9 + 7],
            boxes[i * 9 + 8]
        );
        res.push_back(Bb);
    }
    nms_cpu(res, mParams.nmsIoUThreshold, postNMSBoxes, mParams.preNMSTopN);
    printf("Detected %d objects\n", int(postNMSBoxes.size()));
    printf("Class Name, x, y, z, dx, dy, dz, yaw, score\n");
    printf("===========================================\n");
    for(int i=0; i<postNMSBoxes.size(); i++) {
        printf("%s, %f, %f, %f, %f, %f, %f, %f, %f\n",
        mParams.classNames[postNMSBoxes[i].id].c_str(), postNMSBoxes[i].x,
        postNMSBoxes[i].y, postNMSBoxes[i].z, postNMSBoxes[i].l, postNMSBoxes[i].w,
        postNMSBoxes[i].h, postNMSBoxes[i].rt, postNMSBoxes[i].score);
    }
    return true;
}

SamplePointPillarsParams initializeSampleParams(const samplesCommon::Args& args)
{
    SamplePointPillarsParams params;
    params.modelFile = std::string{"pointpillars.onnx"};
    params.dataFile = std::string{"000001_fov.bin"};
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/pointpillars/");
        params.dataDirs.push_back("data/samples/pointpillars/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.classNames.push_back("car");
    params.classNames.push_back("pedestrian");
    params.classNames.push_back("cyclist");
    params.maxPointsNum = 25000;
    params.numPointFeatures = 4;
    params.batchSize = 1;
    params.nmsIoUThreshold = 0.01;
    params.preNMSTopN = 4096;
    params.doProfile = false;
    params.fp16 = args.runInFp16;
    params.inputTensorNames.push_back("points");
    params.inputTensorNames.push_back("num_points");
    params.outputTensorNames.push_back("output_boxes");
    params.outputTensorNames.push_back("num_boxes");
    params.saveEngine = args.saveEngine;
    params.loadEngine = args.loadEngine;
    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_point_pillars [-h or --help] [-d or --datadir=<path to data directory>]"
              << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "data/samples/pointpillars/ and data/pointpillars/"
              << std::endl;
    std::cout << "--fp16          Specify to run in fp16 mode." << std::endl;
    std::cout << "--int8          Specify to run in int8 mode." << std::endl;
    std::cout << "--saveEngine    Path to save engine." << std::endl;
    std::cout << "--loadEngine    Path to load engine." << std::endl;
}

int32_t main(int32_t argc, char** argv)
{
    samplesCommon::Args args;
    const bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(sampleTest);
    SamplePointPillars sample(initializeSampleParams(args));
    sample::gLogInfo << "Building inference engine for PointPillars" << std::endl;
    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    sample::gLogInfo << "Running inference engine for PointPillars" << std::endl;
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    return sample::gLogger.reportPass(sampleTest);
}
