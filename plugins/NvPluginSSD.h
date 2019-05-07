#ifndef TRT_NV_PLUGIN_SSD_H
#define TRT_NV_PLUGIN_SSD_H

#include "NvInferPlugin.h"
#include "cudnn.h"
#include "ssd.h"
#include <cublas_v2.h>

namespace nvinfer1
{
namespace plugin
{

// Permute {{{
class Permute : public INvPlugin
{
public:
    Permute(Quadruple permuteOrder)
        : permuteOrder(permuteOrder)
    {
    }

    Permute(const void* buffer, size_t length);

    ~Permute() override = default;

    int getNbOutputs() const override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() override;

    void serialize(void* buffer) override;

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override;

    // INvPlugin
    PluginType getPluginType() const override;

    const char* getName() const override;

    void destroy() override;

private:
    bool needPermute{};
    Quadruple permuteOrder{};
    Quadruple oldSteps{};
    Quadruple newSteps{};
};
// Permute }}}
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_NV_PLUGIN_SSD_H
