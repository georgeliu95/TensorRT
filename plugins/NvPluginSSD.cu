#include "NvInfer.h"
#include "NvPluginSSD.h"
#include "checkMacrosPlugin.h"
#include "plugin.h"
#include "ssd.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <vector>

using nvinfer1::Dims;
using nvinfer1::Weights;
using nvinfer1::plugin::Permute;
using std::vector;

using nvinfer1::PluginType;

// Permute {{{
Permute::Permute(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    needPermute = read<bool>(d);
    permuteOrder = read<Quadruple>(d);
    oldSteps = read<Quadruple>(d);
    newSteps = read<Quadruple>(d);
    assert(d == a + length);
}

void Permute::destroy()
{
    delete this;
}

int Permute::getNbOutputs() const
{
    return 1;
}

Dims Permute::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    assert(inputs[0].nbDims == 3);     // target 4D tensors for now
    assert(permuteOrder.data[0] == 0); // do not support permuting batch dimension for now
    for (int i = 0; i < 4; ++i)
    {
        int order = permuteOrder.data[i];
        assert(order < 4);
        if (i > 0)
        {
            assert(std::find(permuteOrder.data, permuteOrder.data + i, order) == permuteOrder.data + i && "There are duplicate orders");
        }
    }
    needPermute = false;
    for (int i = 0; i < 4; ++i)
    {
        if (permuteOrder.data[i] != i)
        {
            needPermute = true;
            break;
        }
    }
    if (needPermute)
        return DimsCHW(inputs[0].d[permuteOrder.data[1] - 1],
                       inputs[0].d[permuteOrder.data[2] - 1],
                       inputs[0].d[permuteOrder.data[3] - 1]);
    else
        return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

int Permute::initialize()
{
    return 0;
}

void Permute::terminate()
{
}

size_t Permute::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int Permute::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    const int count = newSteps.data[0] * batchSize;
    ssdStatus_t status = permuteInference(stream, needPermute, 4, count,
                                          &permuteOrder, &oldSteps, &newSteps,
                                          inputData, outputData);
    assert(status == STATUS_SUCCESS);
    return 0;
}

size_t Permute::getSerializationSize()
{
    // needPermute, Quadruples(permuteOrder, oldSteps, newSteps)
    return sizeof(bool) + sizeof(Quadruple) * 3;
}

void Permute::serialize(void* buffer)
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, needPermute);
    write(d, permuteOrder);
    write(d, oldSteps);
    write(d, newSteps);
    assert(d == a + getSerializationSize());
}

void Permute::configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize)
{
    assert(nbInputs == 1);
    assert(inputDims[0].nbDims == 3); // target 4D tensors for now
    int C = inputDims[0].d[0], H = inputDims[0].d[1], W = inputDims[0].d[2];
    oldSteps = {C * H * W, H * W, W, 1};
    C = outputDims[0].d[0], H = outputDims[0].d[1], W = outputDims[0].d[2];
    newSteps = {C * H * W, H * W, W, 1};
}

PluginType Permute::getPluginType() const { return PluginType::kPERMUTE; }
const char* Permute::getName() const { return "Permute"; }
// Permute }}}
