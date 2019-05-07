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

#ifndef PROFILER_H
#define PROFILER_H

#include "NvInfer.h"
#include "InternalAPI.h"
#include "layerTypeNames.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "logger.h"
#include "common.h"
#include <algorithm>
#include <vector>
#include <string>
#include <cstring>
#include <unordered_map>
#include <iomanip>
#include <ios>
#include <sstream>
#include <tuple>
#include <iostream>
#include <numeric>

std::ostream& operator<<(std::ostream& o, DataType dt)
{
    switch (dt)
    {
    case DataType::kINT32: o << "Int32"; break;
    case DataType::kFLOAT: o << "Float"; break;
    case DataType::kHALF: o << "Half"; break;
    case DataType::kINT8: o << "Int8"; break;
    }
    return o;
}

static void getDeviceInfo(int deviceID, cudaDeviceProp* deviceProp)
{
    CHECK(cudaGetDeviceProperties(deviceProp, deviceID));
}

struct DataRequirements
{
    std::vector<std::pair<nvinfer1::Dims, nvinfer1::DataType>> inputs;
    std::vector<std::pair<nvinfer1::Dims, nvinfer1::DataType>> outputs;
    size_t weightSize;

    size_t getDataSize(int batchSize) const
    {
        size_t res = weightSize;
        for (const auto& elem : inputs)
            res += samplesCommon::volume(elem.first) * samplesCommon::elementSize(elem.second) * batchSize;
        for (const auto& elem : outputs)
            res += samplesCommon::volume(elem.first) * samplesCommon::elementSize(elem.second) * batchSize;
        return res;
    }
};

std::ostream& operator<<(std::ostream& out, const DataRequirements& dataRequirements)
{
    for (int i = 0; i < static_cast<int>(dataRequirements.inputs.size()); ++i)
    {
        if (i > 0)
            out << " ";
        out << dataRequirements.inputs[i].first << " " << dataRequirements.inputs[i].second;
    }
    out << ", ";
    for (int i = 0; i < static_cast<int>(dataRequirements.outputs.size()); ++i)
    {
        if (i > 0)
            out << " ";
        out << dataRequirements.outputs[i].first << " " << dataRequirements.outputs[i].second;
    }
    out << ", ";
    out << std::setw(11) << (float) (dataRequirements.weightSize) << " B";

    return out;
}

struct LayerRequirements
{
    std::pair<int64_t, nvinfer1::DataType> ops; // batch-1 operations, FMA counts as 2 ops
    DataRequirements dataReqs;

    static void reportHeader(std::ostream& out)
    {
        out << std::setw(18) << "Ops per sample"
            << ", " << std::setw(19) << "Input per sample"
            << ", " << std::setw(19) << "Output per sample"
            << ", " << std::setw(13) << "Weights";
    }
};

std::ostream& operator<<(std::ostream& out, const LayerRequirements& layerRequirements)
{
    std::stringstream str;
    str << layerRequirements.ops.first << " ";
    if (layerRequirements.ops.first > 0)
        str << layerRequirements.ops.second;
    else
        str << "    ";
    out << std::setw(18) << str.str();
    out << ", " << layerRequirements.dataReqs;
    return out;
}

struct EngineRequirements
{
    EngineRequirements(const nvinfer1::ICudaEngine& engine)
    {
        layerRequirements.resize(getNbLayers(engine));
        for (size_t i = 0; i < layerRequirements.size(); ++i)
        {
            size_t inputSize = getLayerInputSize(engine, i);
            size_t outputSize = getLayerOutputSize(engine, i);
            std::vector<std::pair<nvinfer1::Dims, nvinfer1::DataType>> inputs(inputSize);
            std::vector<std::pair<nvinfer1::Dims, nvinfer1::DataType>> outputs(outputSize);
            getLayerInputsInfo(engine, inputs.data(), i, inputSize);
            getLayerOutputsInfo(engine, outputs.data(), i, outputSize);
            layerRequirements[i] = std::tuple<std::string, std::string, LayerRequirements>(getLayerName(engine, i), getLayerType(engine, i), LayerRequirements{getLayerOps(engine, i), DataRequirements{inputs, outputs, getLayerWeightsSize(engine, i)}});
        }
    }

    std::vector<std::tuple<std::string, std::string, LayerRequirements>> layerRequirements;
};

std::ostream& operator<<(std::ostream& out, const EngineRequirements& engineRequirements)
{
    int max_layer_name = 0;
    int max_layer_type = 0;
    for (auto layerRequirement : engineRequirements.layerRequirements)
    {
        max_layer_name = std::max(max_layer_name, static_cast<int>(std::get<0>(layerRequirement).size()));
        max_layer_type = std::max(max_layer_type, static_cast<int>(std::get<1>(layerRequirement).size()));
    }
    max_layer_name = std::max(max_layer_name, (int) strlen("Layer name"));
    max_layer_type = std::max(max_layer_type, (int) strlen("Layer type"));

    out << std::setw(max_layer_name) << "Layer name"
        << ", " << std::setw(max_layer_type) << "Layer type"
        << ", ";
    LayerRequirements::reportHeader(out);
    out << std::endl;
    for (const auto& layerRequirementAndName : engineRequirements.layerRequirements)
        out << std::setw(max_layer_name) << std::get<0>(layerRequirementAndName) << ", " << std::setw(max_layer_type) << std::get<1>(layerRequirementAndName) << ", " << std::get<2>(layerRequirementAndName) << std::endl;

    return out;
}

struct CUDADeviceTroughput
{
    CUDADeviceTroughput(
        int deviceID,
        float overrideGPUClocks, // in GHz
        float overrideMemClocks, // in GHz
        float overrideAchievableMemoryBandwidthRatio)
    {
        cudaDeviceProp deviceProp;
        getDeviceInfo(deviceID, &deviceProp);

        eccEnabled = (deviceProp.ECCEnabled != 0);
        gpuClockRate = overrideGPUClocks > 0 ? overrideGPUClocks * 1.0e+9F : deviceProp.clockRate * 1000.0F;
        memClockRate = overrideMemClocks > 0 ? overrideMemClocks * 1.0e+9F : deviceProp.memoryClockRate * 1000.0F;
        achievableMemoryBandwidthRatio = overrideAchievableMemoryBandwidthRatio > 0 ? overrideAchievableMemoryBandwidthRatio : 0.8F;

        fp32flops = gpuClockRate * deviceProp.multiProcessorCount * getCoreCountPerSM(deviceProp.major, deviceProp.minor) * 2.0F;
        eccRatio = (eccEnabled ? ((deviceProp.memoryBusWidth >= 512) ? eccHbmRatio : eccGddrRatio) : 1.0F);
        achievableMemoryBandwidth = memClockRate * 2.0F * (deviceProp.memoryBusWidth / 8) * achievableMemoryBandwidthRatio * eccRatio;
    }

    float getFlops(nvinfer1::DataType t, bool isMMA = false) const
    {
        switch (t)
        {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32:
            return fp32flops;
        case nvinfer1::DataType::kHALF: return fp32flops * (isMMA ? 8.0F : 2.0F);
        case nvinfer1::DataType::kINT8: return fp32flops * (isMMA ? 16.0F : 4.0F);
        }
        assert(0);
        return 0;
    }

public:
    float gpuClockRate; // in Hz
    float memClockRate; // in Hz
    float achievableMemoryBandwidthRatio;
    bool eccEnabled;
    float fp32flops;                 // flops
    float achievableMemoryBandwidth; // B/s
    float eccRatio;

    static constexpr float eccGddrRatio = 0.8F;
    static constexpr float eccHbmRatio = 1.0F;

private:
    // Taken from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    static int getCoreCountPerSM(int majorComputeCapability, int minorComputeCapability)
    {
        int version = majorComputeCapability * 100 + minorComputeCapability;
        if (version < 500)
            return 192;
        else if ((version >= 500 && version < 600) || version == 601 || version == 602)
            return 128;
        else if (version == 600 || version == 700 || version == 702 || version == 705)
            return 64;
        else
            assert(!"Unknown version detected.");
    }
};

std::ostream& operator<<(std::ostream& out, const CUDADeviceTroughput& deviceThroughput)
{
    out << deviceThroughput.fp32flops * 1.0e-12F << " TFLOPS fp32 (@ " << deviceThroughput.gpuClockRate * 1.0e-9F << " GHz), "
        << deviceThroughput.achievableMemoryBandwidth * 1.0e-9F << " GB/s practical achievable mem bw (@ "
        << deviceThroughput.memClockRate * 1.0e-9F << " GHz, assuming "
        << deviceThroughput.achievableMemoryBandwidthRatio << " achievable/theoretical peak ratio";
    if (deviceThroughput.eccEnabled)
        out << ", ECC enabled, assuming another " << deviceThroughput.eccRatio << " ratio";
    out << ")";
    return out;
};

struct SOLProfiler : public nvinfer1::IProfiler
{
    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto it = layerNameToRunTimeVector.find(layerName);
        if (it == layerNameToRunTimeVector.end())
            layerNameToRunTimeVector.insert(make_pair(std::string(layerName), std::vector<float>(1, ms * 1.0e-3F)));
        else
            it->second.push_back(ms * 1.0e-3F);
    }

    float getAverageRuntime(const char* layerName) const
    {
        auto it = layerNameToRunTimeVector.find(layerName);
        if (it == layerNameToRunTimeVector.end())
            return 0.0F;
        else
            return std::accumulate(it->second.begin(), it->second.end(), 0.0F) / (float) (it->second.size());
    }

    float getMedianRuntime(const char* layerName) const
    {
        auto it = layerNameToRunTimeVector.find(layerName);
        if (it == layerNameToRunTimeVector.end())
            return 0.0F;
        else
        {
            std::vector<float> vals = it->second;
            std::sort(vals.begin(), vals.end());
            if (vals.size() % 2 == 1)
                return vals[vals.size() / 2];
            else
                return (vals[vals.size() / 2] + vals[vals.size() / 2 + 1]) * 0.5F;
        }
    }

    std::unordered_map<std::string, std::vector<float>> layerNameToRunTimeVector;
};

void reportPerLayerSOLAnalysis(
    const SOLProfiler& profiler,
    const CUDADeviceTroughput& deviceThroughput,
    const EngineRequirements& engineRequirement,
    int batchSize,
    bool showEngineReqs)
{
    float totalActualRuntime = 0;
    int max_layer_name = 0;
    int max_layer_type = 0;
    for (const auto& layerRequirement : engineRequirement.layerRequirements)
    {
        max_layer_name = std::max(max_layer_name, static_cast<int>(std::get<0>(layerRequirement).size()));
        max_layer_type = std::max(max_layer_type, static_cast<int>(std::get<1>(layerRequirement).size()));
        totalActualRuntime += profiler.getMedianRuntime(std::get<0>(layerRequirement).c_str());
    }
    max_layer_name = std::max(max_layer_name, (int) strlen("Layer name"));
    max_layer_type = std::max(max_layer_type, (int) strlen("Layer type"));

    gLogInfo << std::setw(max_layer_name) << "Layer name"
             << ", " << std::setw(max_layer_type) << "Layer type"
             << ", "
             << "% Runtime"
             << ", "
             << " % SOL"
             << ", "
             << "  Lim"
             << ", "
             << "     Runtime"
             << ", "
             << "FLOPs runtime"
             << ", "
             << " Mem runtime";
    if (showEngineReqs)
    {
        gLogInfo << ", ";
        LayerRequirements::reportHeader(gLogInfo);
    }
    gLogInfo << std::endl;

    auto old_settings = gLogInfo.flags();
    auto old_precision = gLogInfo.precision();
    for (const auto& layerRequirementAndName : engineRequirement.layerRequirements)
    {
        const auto& layerRequirement = std::get<2>(layerRequirementAndName);
        bool isMMA = std::get<1>(layerRequirementAndName).find("MMA") != std::string::npos;
        float flopsLimitedRuntime = layerRequirement.ops.first * batchSize / deviceThroughput.getFlops(layerRequirement.ops.second, isMMA);
        float memLimitedRuntime = layerRequirement.dataReqs.getDataSize(batchSize) / deviceThroughput.achievableMemoryBandwidth;
        float actualRuntime = profiler.getMedianRuntime(std::get<0>(layerRequirementAndName).c_str());
        float solRuntime = std::max(flopsLimitedRuntime, memLimitedRuntime);

        gLogInfo << std::setw(max_layer_name) << std::get<0>(layerRequirementAndName) << ", ";
        gLogInfo << std::setw(max_layer_type) << std::get<1>(layerRequirementAndName) << ", ";
        gLogInfo << std::setw(8) << std::fixed << std::setprecision(1) << (actualRuntime / totalActualRuntime * 100.0F) << "%"
                 << ", ";
        gLogInfo.flags(old_settings);
        gLogInfo.precision(old_precision);
        gLogInfo << std::setw(5) << std::fixed << std::setprecision(1) << (solRuntime / actualRuntime * 100.0F) << "%"
                 << ", ";
        gLogInfo.flags(old_settings);
        gLogInfo.precision(old_precision);
        gLogInfo << std::setw(5) << ((memLimitedRuntime > flopsLimitedRuntime) ? "mem" : "flops") << ", ";
        gLogInfo << std::setw(9) << std::fixed << std::setprecision(4) << actualRuntime * 1000.0F << " ms, ";
        gLogInfo.flags(old_settings);
        gLogInfo.precision(old_precision);
        gLogInfo << std::setw(10) << std::fixed << std::setprecision(4) << flopsLimitedRuntime * 1000.0F << " ms, ";
        gLogInfo.flags(old_settings);
        gLogInfo.precision(old_precision);
        gLogInfo << std::setw(9) << std::fixed << std::setprecision(4) << memLimitedRuntime * 1000.0F << " ms";
        gLogInfo.flags(old_settings);
        gLogInfo.precision(old_precision);

        if (showEngineReqs)
            gLogInfo << ", " << layerRequirement;

        gLogInfo << std::endl;
    }
}

float estimateRuntime(
    const CUDADeviceTroughput& deviceThroughput,
    const EngineRequirements& engineRequirement,
    int batchSize,
    int streams,
    float flopsEfficiency)
{
    float res = 0;
    for (const auto& layerRequirementAndName : engineRequirement.layerRequirements)
    {
        const auto& layerRequirement = std::get<2>(layerRequirementAndName);
        float flopsRuntime = layerRequirement.ops.first * batchSize / deviceThroughput.getFlops(layerRequirement.ops.second) / flopsEfficiency * streams;
        float memLimitedRuntime = layerRequirement.dataReqs.getDataSize(batchSize) / deviceThroughput.achievableMemoryBandwidth * streams;
        res += std::max(flopsRuntime, memLimitedRuntime);
    }
    return res;
}

float getNetworkFlopsEfficiency(
    const CUDADeviceTroughput& deviceThroughput,
    const EngineRequirements& engineRequirement,
    int batchSize,
    int streams,
    float runTime)
{
    std::pair<float, float> lowerBound(0.0001F, estimateRuntime(deviceThroughput, engineRequirement, batchSize, streams, 0.0001F));
    if (lowerBound.second < runTime)
        return lowerBound.first;
    std::pair<float, float> upperBound(10.0F, estimateRuntime(deviceThroughput, engineRequirement, batchSize, streams, 10.0F));
    if (upperBound.second > runTime)
        return upperBound.first;
    while (upperBound.first - lowerBound.first > 0.001F)
    {
        std::pair<float, float> middle((lowerBound.first + upperBound.first) * 0.5F, estimateRuntime(deviceThroughput, engineRequirement, batchSize, streams, (lowerBound.first + upperBound.first) * 0.5F));
        if (middle.second > runTime)
            lowerBound = middle;
        else
            upperBound = middle;
    }
    return (lowerBound.first + upperBound.first) * 0.5F;
}

#endif
