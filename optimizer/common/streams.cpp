#include "common/parameterPrinter.h"
#include "common/fillOpParameters.h"
#include "common/sliceOpParameters.h"
#include "runtime/common/streams.h"
#include "core/layerParameters.h"
#include "core/layerTypeNames.h"
namespace nvinfer1
{
std::ostream& printParameters(std::ostream& o, nvinfer1::Quantization const& p, bool const isJSON, bool const isFirstParam);
std::ostream& printParameters(std::ostream& o, nvinfer1::Quantizations const& p, bool const isJSON, bool const isFirstParam);
std::ostream& printParameters(
    std::ostream& o, builder::FillOpParameters const& p, bool const isJSON /* = false */)
{
    printParameterValuePair(o, "ParameterType", "Fill", true, false, isJSON);
    printParameterValuePair(o, "FillOperation", p.op, true, false, isJSON);
    printParameterValuePair(o, "Dimentions", p.dimensions, false, false, isJSON);
    printParameterValuePair(o, "Alpha", p.alpha, false, false, isJSON);
    printParameterValuePair(o, "Beta", p.beta, false, false, isJSON);
    printParametersAsObject(o, "Quantizations", p.quantizations, false, isJSON);
    return o;
}
std::ostream& printParameters(
    std::ostream& o, builder::SliceOpParameters const& p, bool const isJSON /* = false */)
{
    printParameterValuePair(o, "ParameterType", "Slice", true, false, isJSON);
    printParameterValuePair(o, "Mode", p.mode, true, false, isJSON);
    printParameterValuePair(o, "Start", p.start, false, false, isJSON);
    printParameterValuePair(o, "Size", p.size, false, false, isJSON);
    printParameterValuePair(o, "Stride", p.stride, false, false, isJSON);
    printParameterValuePair(o, "NegativeInfinityPadding", p.negativeInfinityPadding, false, false, isJSON);
    printParametersAsObject(o, "Quantizations", p.quantizations, false, isJSON);
    return o;
}

std::ostream& operator<<(std::ostream& o, nvinfer1::builder::FillOpParameters const& p)
{
    return printParameters(o, p, false);
}

std::ostream& operator<<(std::ostream& o, nvinfer1::builder::SliceOpParameters const& p)
{
    return printParameters(o, p, false);
}
}
