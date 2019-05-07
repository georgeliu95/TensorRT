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
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <string>
#include <vector>

#include "weight_utils.h"
#include "error_util.h"
#include "logger.h"
#include "common.h"

using namespace nvinfer1;

namespace RNNDataUtil
{

// We have the data files located in a specific directory. This
// searches for that directory format from the current directory.
std::string locateFile2(const std::string& dataDir, const std::string& input)
{
    std::vector<std::string> dirs{std::string(dataDir) + "/"};
    return locateFile(input, dirs);
}

template <nvinfer1::DataType nvInferType>
std::map<std::string, Weights> loadWeights(const std::string& dataDir, const std::string& weightsFileName)
{
    std::string file = locateFile2(dataDir, weightsFileName);
    std::map<std::string, Weights> weightMap;
    std::ifstream input(file);
    if (!input.is_open())
        FatalError("Can't open " + file);
    std::string weightsName;
    std::string line;
    std::string binDir;
    // parse descriptor file:
    std::getline(input, binDir);
    gLogInfo << "Loading weights" << std::endl;
    while (std::getline(input, line))
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::istringstream iss(line);
        iss >> weightsName >> type >> size;
        wt.type = static_cast<DataType>(type);

        std::streampos weightsSize;
        std::string fileName = locateFile2(dataDir, binDir + '/' + weightsName);
        std::ifstream weightsFile(fileName, std::ios::in | std::ios::binary | std::ios::ate);
        gLogInfo << "\"" << fileName << "\"" << std::endl;
        if (!weightsFile.is_open())
            FatalError("Can't open " + weightsName);
        weightsSize = weightsFile.tellg();
        weightsFile.seekg(0, std::ios::beg);
        if (wt.type == DataType::kFLOAT)
        {
            uint32_t bytes = sizeof(uint32_t) * size;
            uint32_t* val = reinterpret_cast<uint32_t*>(malloc(bytes));
            if (static_cast<size_t>(weightsSize) != bytes)
                FatalError(weightsName + "  size is not consistent");
            weightsFile.read((char*) val, weightsSize);
            // need to cast data
            if (nvInferType == DataType::kHALF)
            {
                bytes = sizeof(half1) * size;
                half1* valhf = reinterpret_cast<half1*>(malloc(bytes));
                Convert<half1> fromFloat;
                for (uint32_t i = 0; i < size; i++)
                {
                    valhf[i] = fromFloat(*((float*) val + i));
                }
                wt.values = valhf;
                wt.type = static_cast<DataType>(nvInferType);
                free(val);
            }
            else
            {
                wt.values = val;
            }
        }
        else if (wt.type == DataType::kHALF)
        {
            if (nvInferType != DataType::kHALF)
            {
                FatalError("Not Implemented");
            }
            uint32_t bytes = sizeof(uint16_t) * size;
            uint16_t* val = reinterpret_cast<uint16_t*>(malloc(bytes));
            weightsFile.read((char*) val, bytes);
            wt.values = val;
        }
        weightsFile.close();

        wt.count = size;
        // use erase to remove extension '.bin'
        weightMap[weightsName.erase(weightsName.size() - 4, 4)] = wt;
    }
    return weightMap;
}

// TensorFlow weight parameters for BasicLSTMCell
// are formatted as:
// Each [WR][icfo] is hiddenSize sequential elements.
// CellN  Row 0: WiT, WcT, WfT, WoT
// CellN  Row 1: WiT, WcT, WfT, WoT
// ...
// CellN RowM-1: WiT, WcT, WfT, WoT
// CellN RowM+0: RiT, RcT, RfT, RoT
// CellN RowM+1: RiT, RcT, RfT, RoT
// ...
// CellNRow(M+P)-1: RiT, RcT, RfT, RoT
// M - data size
// P - projection size
// TensorRT expects the format to laid out in memory:
// CellN: Wf, Wi, Wc, Wo, Rf, Ri, Rc, Ro

// For the purpose of implementing LSTMP all W and R weights become weights from W
// CellN: Wf, Rf, Wi, Ri, Wc, Rc, Wo, Ro, Empty states
template <class value_type>
Weights convertRNNWeights(Weights input, uint32_t DATA_SIZE, uint32_t PROJECTED_SIZE, uint32_t CELL_SIZE, uint32_t LAYER_COUNT)
{
    const value_type* iptr = static_cast<const value_type*>(input.values);
    Convert<value_type> fromFloat;
    value_type zero = fromFloat(0);
    uint32_t u_count = CELL_SIZE * CELL_SIZE; // We put everything into input, but still need to allcoate U
    value_type* ptr = static_cast<value_type*>(malloc(sizeof(value_type) * (input.count + 4 * u_count * LAYER_COUNT)));
    std::fill(ptr, ptr + (input.count + 4 * u_count * LAYER_COUNT), zero);
    uint32_t indir[4]{1, 2, 0, 3};
    // First lets until the tensorflow weights
    for (uint32_t z = 0; z < LAYER_COUNT; ++z)
        for (uint32_t r = 0; r < 2; ++r)
            for (uint32_t x = 0; x < 4; ++x)
                for (uint32_t w = 0; w < CELL_SIZE; ++w)
                    for (uint32_t q = 0; q < PROJECTED_SIZE; ++q)
                    {
                        // we assume PROJECTED_SIZE == DATA_SIZE
                        uint32_t h_2 = CELL_SIZE * PROJECTED_SIZE;
                        uint32_t subMatIdx = (z * 2 + r) * h_2 * 4;
                        uint32_t srcIdx = w
                            + x * CELL_SIZE
                            + q * CELL_SIZE * 4
                            + subMatIdx;
                        //  Workaround for LSTMP: Increase the offset here to take into account empty U state
                        uint32_t dstIdx = q
                            + r * PROJECTED_SIZE
                            + w * 2 * PROJECTED_SIZE
                            + indir[x] * (2 * PROJECTED_SIZE * CELL_SIZE)
                            + z * 4 * (2 * PROJECTED_SIZE * CELL_SIZE + CELL_SIZE * CELL_SIZE);
                        ptr[dstIdx] = iptr[srcIdx];
                    }
    return Weights{input.type, ptr, input.count + 4 * u_count * LAYER_COUNT};
}

// Set weights on RNNv2 layer
// ptr: Pointer to weights buffer on host
template <nvinfer1::DataType nvInferType>
void setRNNLayerWeights(
    nvinfer1::IRNNv2Layer* rnnLayer,
    const void* vPtr,
    uint32_t inputSize,
    uint32_t projectionSize,
    uint32_t hiddenSize,
    uint32_t layerIdx)
{
    typedef typename InferTypeToValue<nvInferType>::value_type value_type;
    value_type* wPtr = (value_type*) vPtr;

    RNNGateType gateTypes[4] = {
        RNNGateType::kFORGET,
        RNNGateType::kINPUT,
        RNNGateType::kCELL,
        RNNGateType::kOUTPUT};

    const uint32_t subMatrixSizes[2] = {
        ((inputSize + projectionSize) * hiddenSize), // Size of W matrix for each gate
        (hiddenSize * hiddenSize)                    // Size of R matrix for each gate
    };

    // Iterate W (j=0) and R (j=1) matrices
    for (int j = 0; j < 2; ++j)
    {
        const bool isW = (j == 0);

        // Iterate 4 gates in FICO order
        for (int i = 0; i < 4; ++i)
        {
            Weights weights;
            weights.type = nvInferType;
            weights.count = subMatrixSizes[j];
            weights.values = (void*) wPtr;

            rnnLayer->setWeightsForGate(
                layerIdx,
                gateTypes[i],
                isW,
                weights);

            wPtr += subMatrixSizes[j];
        }
    }
}

// Set biases on RNNv2 layer
// vPtr: Pointer to biases buffer on host
template <nvinfer1::DataType nvInferType>
void setRNNLayerBiases(
    nvinfer1::IRNNv2Layer* rnnLayer,
    const void* vPtr,
    uint32_t hiddenSize,
    uint32_t layerIdx)
{
    typedef typename InferTypeToValue<nvInferType>::value_type value_type;
    value_type* wPtr = (value_type*) vPtr;

    RNNGateType gateTypes[4] = {
        RNNGateType::kFORGET,
        RNNGateType::kINPUT,
        RNNGateType::kCELL,
        RNNGateType::kOUTPUT};

    const uint32_t subMatrixSizes[2] = {
        hiddenSize, // Size of W bias matrix for each gate
        hiddenSize  // Size of R bias matrix for each gate
    };

    // Iterate W (j=0) and R (j=1) matrices
    for (int j = 0; j < 2; ++j)
    {
        const bool isW = (j == 0);

        // Iterate 4 gates in FICO order
        for (int i = 0; i < 4; ++i)
        {
            Weights weights;
            weights.type = nvInferType;
            weights.count = subMatrixSizes[j];
            weights.values = (void*) wPtr;

            rnnLayer->setBiasForGate(
                layerIdx,
                gateTypes[i],
                isW,
                weights);

            wPtr += subMatrixSizes[j];
        }
    }
}

// TensorFlow bias parameters for BasicLSTMCell
// are formatted as:
// CellN: Bi, Bc, Bf, Bo
//
// TensorRT expects the format to be:
// CellN: Wf, Wi, Wc, Wo, Rf, Ri, Rc, Ro
//
// Since tensorflow already combines U and W,
// we double the size and set all of U to zero.
template <class value_type>
Weights convertRNNBias(Weights input, uint32_t DATA_SIZE, uint32_t PROJECTED_SIZE, uint32_t CELL_SIZE, uint32_t LAYER_COUNT)
{
    Convert<value_type> fromFloat;
    Convert<float> toFloat;

    value_type* ptr = static_cast<value_type*>(malloc(sizeof(value_type) * input.count * 2));
    value_type zero = fromFloat(0);
    std::fill(ptr, ptr + input.count * 2, zero);
    const value_type* iptr = static_cast<const value_type*>(input.values);
    int indir[4]{1, 2, 0, 3};

    for (uint32_t z = 0, y = 0; z < LAYER_COUNT; ++z)
        for (uint32_t x = 0; x < 4; ++x, ++y)
            std::copy(iptr + y * CELL_SIZE, iptr + (y + 1) * CELL_SIZE, ptr + (z * 8 + indir[x]) * CELL_SIZE);

    // Add 1 to f bias to be consistant with the Tensorflow model
    for (uint32_t z = 0; z < LAYER_COUNT; ++z)
        for (uint32_t i = 0; i < CELL_SIZE; i++)
        {
            float val = toFloat(*(ptr + 0 * CELL_SIZE + z * 8 * CELL_SIZE + i));
            val += 1.0f;
            *(ptr + 0 * CELL_SIZE + z * 8 * CELL_SIZE + i) = fromFloat(val);
        }
    return Weights{input.type, ptr, input.count * 2};
}

// The fully connected weights from tensorflow are transposed compared to
// the order that tensorRT expects them to be in.
template <class value_type>
Weights transposeFCWeights(Weights input, uint32_t nRows, uint32_t nCols, uint32_t nLayers = 1)
{
    value_type* ptr = static_cast<value_type*>(malloc(sizeof(value_type) * input.count));
    const value_type* iptr = static_cast<const value_type*>(input.values);
    assert(input.count == nLayers * nCols * nRows);
    for (uint32_t i = 0; i < nLayers; i++)
        for (uint32_t z = 0; z < nCols; ++z)
            for (uint32_t x = 0; x < nRows; ++x)
                ptr[x * nCols + z + i * nCols * nRows] = iptr[z * nRows + x + i * nCols * nRows];
    return Weights{input.type, ptr, input.count};
}

template <nvinfer1::DataType nvInferType>
void preprocessWeights(std::map<std::string, Weights>& weightMap, uint32_t DATA_SIZE, uint32_t PROJECTED_SIZE, uint32_t CELL_SIZE, uint32_t LAYER_COUNT)
{
    auto tfwts = weightMap["rnnweight"];
    typedef typename InferTypeToValue<nvInferType>::value_type value_type;
    Weights rnnwts = convertRNNWeights<value_type>(tfwts, DATA_SIZE, PROJECTED_SIZE, CELL_SIZE, LAYER_COUNT);
    auto tfbias = weightMap["rnnbias"];
    Weights rnnbias = convertRNNBias<value_type>(tfbias, DATA_SIZE, PROJECTED_SIZE, CELL_SIZE, LAYER_COUNT);
    auto tfwtsp = weightMap["rnnweightp"];
    auto rnnwtsp = transposeFCWeights<value_type>(tfwtsp, PROJECTED_SIZE, CELL_SIZE, LAYER_COUNT);
    // LOGITS: transpose is not needed
    // Store the transformed weights in the weight map for later use.
    weightMap["rnnweightp2"] = rnnwtsp;
    weightMap["rnnweight2"] = rnnwts;
    weightMap["rnnbias2"] = rnnbias;
}

template std::map<std::string, Weights> loadWeights<nvinfer1::DataType::kFLOAT>(const std::string& dataDir, const std::string& weightsFileName);
template std::map<std::string, Weights> loadWeights<nvinfer1::DataType::kHALF>(const std::string& dataDir, const std::string& weightsFileName);

template void preprocessWeights<nvinfer1::DataType::kFLOAT>(std::map<std::string, Weights>& weightMap, uint32_t DATA_SIZE,
                                                            uint32_t PROJECTED_SIZE, uint32_t CELL_SIZE, uint32_t LAYER_COUNT);
template void preprocessWeights<nvinfer1::DataType::kHALF>(std::map<std::string, Weights>& weightMap, uint32_t DATA_SIZE,
                                                           uint32_t PROJECTED_SIZE, uint32_t CELL_SIZE, uint32_t LAYER_COUNT);

template void setRNNLayerWeights<nvinfer1::DataType::kFLOAT>(
    nvinfer1::IRNNv2Layer* rnnLayer,
    const void* vPtr,
    uint32_t inputSize,
    uint32_t projectionSize,
    uint32_t hiddenSize,
    uint32_t layerIdx);
template void setRNNLayerWeights<nvinfer1::DataType::kHALF>(
    nvinfer1::IRNNv2Layer* rnnLayer,
    const void* vPtr,
    uint32_t inputSize,
    uint32_t projectionSize,
    uint32_t hiddenSize,
    uint32_t layerIdx);

template void setRNNLayerBiases<nvinfer1::DataType::kFLOAT>(
    nvinfer1::IRNNv2Layer* rnnLayer,
    const void* vPtr,
    uint32_t hiddenSize,
    uint32_t layerIdx);
template void setRNNLayerBiases<nvinfer1::DataType::kHALF>(
    nvinfer1::IRNNv2Layer* rnnLayer,
    const void* vPtr,
    uint32_t hiddenSize,
    uint32_t layerIdx);
} // end namespace RNNDataUtil
