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

#ifndef NV_INFER_H
#define NV_INFER_H

#include "NvInferRTExt.h"

//!
//! \mainpage
//!
//! This is the API documentation for the NVIDIA TensorRT library. It provides information on individual
//! functions, classes and methods. Use the index on the left to navigate the documentation.
//!
//! Please see the accompanying user guide and samples for higher-level information and general advice on
//! using TensorRT.
//
//! TensorRT Versioning follows Semantic Versioning Guidelines specified here: https://semver.org/
//!

//!
//! \file NvInfer.h
//!
//! This is the top-level API file for TensorRT.
//!

//!
//! \namespace nvinfer1
//!
//! \brief The TensorRT API version 1 namespace.
//!
namespace nvinfer1
{

//!
//! \class Dims2
//! \brief Descriptor for two-dimensional data.
//!
class Dims2 : public Dims
{
public:
    //!
    //! \brief Construct an empty Dims2 object.
    //!
    Dims2()
    {
        nbDims = 2;
        d[0] = d[1] = 0;
    }

    //!
    //! \brief Construct a Dims2 from 2 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //!
    Dims2(int d0, int d1)
    {
        nbDims = 2;
        d[0] = d0;
        d[1] = d1;
    }
};

//!
//! \class DimsHW
//! \brief Descriptor for two-dimensional spatial data.
//!
class DimsHW : public Dims2
{
public:
    //!
    //! \brief Construct an empty DimsHW object.
    //!
    DimsHW()
        : Dims2()
    {
        type[0] = type[1] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Construct a DimsHW given height and width.
    //!
    //! \param Height the height of the data
    //! \param Width the width of the data
    //!
    DimsHW(int height, int width)
        : Dims2(height, width)
    {
        type[0] = type[1] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int& h() { return d[0]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int h() const { return d[0]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int& w() { return d[1]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int w() const { return d[1]; }
};

//!
//! \class Dims3
//! \brief Descriptor for three-dimensional data.
//!
class Dims3 : public Dims
{
public:
    //!
    //! \brief Construct an empty Dims3 object.
    //!
    Dims3()
    {
        nbDims = 3;
        d[0] = d[1] = d[2] = 0;
    }

    //!
    //! \brief Construct a Dims3 from 3 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //! \param d2 The third element.
    //!
    Dims3(int d0, int d1, int d2)
    {
        nbDims = 3;
        d[0] = d0;
        d[1] = d1;
        d[2] = d2;
    }
};

//!
//! \class DimsCHW
//! \brief Descriptor for data with one channel dimension and two spatial dimensions.
//!
//! \deprecated DimsCHW will be removed in a future version of TensorRT, use Dims3 instead.
//!
class TRT_DEPRECATED DimsCHW : public Dims3
{
public:
    //!
    //! \brief Construct an empty DimsCHW object.
    //!
    DimsCHW()
        : Dims3()
    {
        type[0] = DimensionType::kCHANNEL;
        type[1] = type[2] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Construct a DimsCHW given channel count, height and width.
    //!
    //! \param channels The channel count.
    //! \param height The height of the data.
    //! \param width The width of the data.
    //!
    DimsCHW(int channels, int height, int width)
        : Dims3(channels, height, width)
    {
        type[0] = DimensionType::kCHANNEL;
        type[1] = type[2] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Get the channel count.
    //!
    //! \return The channel count.
    //!
    int& c() { return d[0]; }

    //!
    //! \brief Get the channel count.
    //!
    //! \return The channel count.
    //!
    int c() const { return d[0]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int& h() { return d[1]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int h() const { return d[1]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int& w() { return d[2]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int w() const { return d[2]; }
};

//!
//! \class Dims4
//! \brief Descriptor for four-dimensional data.
//!
class Dims4 : public Dims
{
public:
    //!
    //! \brief Construct an empty Dims2 object.
    //!
    Dims4()
    {
        nbDims = 4;
        d[0] = d[1] = d[2] = d[3] = 0;
    }

    //!
    //! \brief Construct a Dims4 from 4 elements.
    //!
    //! \param d0 The first element.
    //! \param d1 The second element.
    //! \param d2 The third element.
    //! \param d3 The fourth element.
    //!
    Dims4(int d0, int d1, int d2, int d3)
    {
        nbDims = 4;
        d[0] = d0;
        d[1] = d1;
        d[2] = d2;
        d[3] = d3;
    }
};

//!
//! \class DimsNCHW
//! \brief Descriptor for data with one index dimension, one channel dimension and two spatial dimensions.
//!
//! \deprecated DimsNCHW will be removed in a future version of TensorRT, use Dims instead.
//!
class TRT_DEPRECATED DimsNCHW : public Dims4
{
public:
    //!
    //! \brief Construct an empty DimsNCHW object.
    //!
    DimsNCHW()
        : Dims4()
    {
        type[0] = DimensionType::kINDEX;
        type[1] = DimensionType::kCHANNEL;
        type[2] = type[3] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Construct a DimsNCHW given batch size, channel count, height and width.
    //!
    //! \param batchSize The batch size (commonly denoted N).
    //! \param channels The channel count.
    //! \param height The height of the data.
    //! \param width The width of the data.
    //!
    DimsNCHW(int batchSize, int channels, int height, int width)
        : Dims4(batchSize, channels, height, width)
    {
        type[0] = DimensionType::kINDEX;
        type[1] = DimensionType::kCHANNEL;
        type[2] = type[3] = DimensionType::kSPATIAL;
    }

    //!
    //! \brief Get the index count.
    //!
    //! \return The index count.
    //!
    int& n() { return d[0]; }

    //!
    //! \brief Get the index count.
    //!
    //! \return The index count.
    //!
    int n() const { return d[0]; }

    //!
    //! \brief Get the channel count.
    //!
    //! \return The channel count.
    //!
    int& c() { return d[1]; }

    //!
    //! \brief Get the channel count.
    //!
    //! \return The channel count.
    //!
    int c() const { return d[1]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int& h() { return d[2]; }

    //!
    //! \brief Get the height.
    //!
    //! \return The height.
    //!
    int h() const { return d[2]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int& w() { return d[3]; }

    //!
    //! \brief Get the width.
    //!
    //! \return The width.
    //!
    int w() const { return d[3]; }
};

//!
//! \enum LayerType
//!
//! \brief The type values of layer classes.
//!
//! \see ILayer::getType()
//!
enum class LayerType : int
{
    kCONVOLUTION = 0,      //!< Convolution layer.
    kFULLY_CONNECTED = 1,  //!< Fully connected layer.
    kACTIVATION = 2,       //!< Activation layer.
    kPOOLING = 3,          //!< Pooling layer.
    kLRN = 4,              //!< LRN layer.
    kSCALE = 5,            //!< Scale layer.
    kSOFTMAX = 6,          //!< SoftMax layer.
    kDECONVOLUTION = 7,    //!< Deconvolution layer.
    kCONCATENATION = 8,    //!< Concatenation layer.
    kELEMENTWISE = 9,      //!< Elementwise layer.
    kPLUGIN = 10,          //!< Plugin layer.
    kRNN = 11,             //!< RNN layer.
    kUNARY = 12,           //!< UnaryOp operation Layer.
    kPADDING = 13,         //!< Padding layer.
    kSHUFFLE = 14,         //!< Shuffle layer.
    kREDUCE = 15,          //!< Reduce layer.
    kTOPK = 16,            //!< TopK layer.
    kGATHER = 17,          //!< Gather layer.
    kMATRIX_MULTIPLY = 18, //!< Matrix multiply layer.
    kRAGGED_SOFTMAX = 19,  //!< Ragged softmax layer.
    kCONSTANT = 20,        //!< Constant layer.
    kRNN_V2 = 21,          //!< RNNv2 layer.
    kIDENTITY = 22,        //!< Identity layer.
    kPLUGIN_V2 = 23,       //!< PluginV2 layer.
    kSLICE = 24,           //!< Slice layer.
    kSHAPE = 25,           //!< Shape layer.
    kPARAMETRIC_RELU = 26, //!< Parametric ReLU layer.
    kRESIZE = 27           //!< Resize Layer.
};

template <>
constexpr inline int EnumMax<LayerType>()
{
    return 28;
} //!< Maximum number of elements in LayerType enum. \see LayerType

//!
//! \class ITensor
//!
//! \brief A tensor in a network definition.
//!
//! to remove a tensor from a network definition, use INetworkDefinition::removeTensor()
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ITensor
{
public:
    //!
    //! \brief Set the tensor name.
    //!
    //! For a network input, the name is assigned by the application. For tensors which are layer outputs,
    //! a default name is assigned consisting of the layer name followed by the index of the output in brackets.
    //!
    //! This method copies the name string.
    //!
    //! \param name The name.
    //!
    //! \see getName()
    //!
    virtual void setName(const char* name) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the tensor name.
    //!
    //! \return The name, as a pointer to a NULL-terminated character sequence.
    //!
    //! \see setName()
    //!
    virtual const char* getName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the dimensions of a tensor.
    //!
    //! For a network input the name is assigned by the application. For a network output it is computed based on
    //! the layer parameters and the inputs to the layer. If a tensor size or a parameter is modified in the network,
    //! the dimensions of all dependent tensors will be recomputed.
    //!
    //! This call is only legal for network input tensors, since the dimensions of layer output tensors are inferred
    //! based on layer inputs and parameters.
    //!
    //! \param dimensions The dimensions of the tensor.
    //!
    //! \see getDimensions()
    //!
    virtual void setDimensions(Dims dimensions) TRTNOEXCEPT = 0; // only valid for input tensors

    //!
    //! \brief Get the dimensions of a tensor.
    //!
    //! \return The dimensions of the tensor.
    //!
    //! \warning getDimensions() returns a -1 for dimensions that are derived from a wildcard dimension.
    //! \see setDimensions()
    //!
    virtual Dims getDimensions() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the data type of a tensor.
    //!
    //! \param type The data type of the tensor.
    //!
    //! The type is unchanged if the type is
    //! invalid for the given tensor.
    //!
    //! \see getType()
    //!
    virtual void setType(DataType type) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the data type of a tensor.
    //!
    //! \return The data type of the tensor.
    //!
    //! \see setType()
    //!
    virtual DataType getType() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set dynamic range for the tensor
    //!
    //! Currently, only symmetric ranges are supported.
    //! Therefore, the larger of the absolute values of the provided bounds is used.
    //!
    //! \return Whether the dynamic range was set successfully.
    //!
    //! Requires that min and max be finite, and min <= max.
    //!
    virtual bool setDynamicRange(float min, float max) TRTNOEXCEPT = 0;

    //!
    //! \brief Get dynamic range for the tensor
    //!
    //! \return maximal absolute value of the dynamic range, -1.0f if no dynamic range is set.
    //!
    //! \deprecated This interface is superceded by getDynamicRangeMin and getDynamicRangeMax.
    //!
    TRT_DEPRECATED virtual float getDynamicRange() const TRTNOEXCEPT = 0;

    //!
    //! \brief Whether the tensor is a network input.
    //!
    virtual bool isNetworkInput() const TRTNOEXCEPT = 0;

    //!
    //! \brief Whether the tensor is a network output.
    //!
    virtual bool isNetworkOutput() const TRTNOEXCEPT = 0;

protected:
    virtual ~ITensor() {}

public:
    //!
    //! \brief Set whether to enable broadcast of tensor across the batch.
    //!
    //! When a tensor is broadcast across a batch, it has the same value for every member in the batch.
    //! Memory is only allocated once for the single member.
    //!
    //! This method is only valid for network input tensors, since the flags of layer output tensors are inferred based
    //! on layer inputs and parameters.
    //! If this state is modified for a tensor in the network, the states of all dependent tensors will be recomputed.
    //! If the tensor is for an explicit batch network, then this function does nothing.
    //!
    //! \param broadcastAcrossBatch Whether to enable broadcast of tensor across the batch.
    //!
    //! \see getBroadcastAcrossBatch()
    //!
    virtual void setBroadcastAcrossBatch(bool broadcastAcrossBatch) TRTNOEXCEPT = 0;

    //!
    //! \brief Check if tensor is broadcast across the batch.
    //!
    //! When a tensor is broadcast across a batch, it has the same value for every member in the batch.
    //! Memory is only allocated once for the single member. If the network is in explicit batch mode,
    //! this function returns true if the leading dimension is 1.
    //!
    //! \return True if tensor is broadcast across the batch, false otherwise.
    //!
    //! \see setBroadcastAcrossBatch()
    //!
    virtual bool getBroadcastAcrossBatch() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the storage location of a tensor.
    //! \return The location of tensor data.
    //! \see setLocation()
    //!
    virtual TensorLocation getLocation() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the storage location of a tensor
    //! \param location the location of tensor data
    //!
    //! Only input tensors for storing sequence lengths for RNNv2 are supported.
    //! Using host storage for layers that do not support it will generate
    //! errors at build time.
    //!
    //! \see getLocation()
    //!
    virtual void setLocation(TensorLocation location) TRTNOEXCEPT = 0;

    //!
    //! \brief Query whether dynamic range is set.
    //!
    //! \return True if dynamic range is set, false otherwise.
    //!
    virtual bool dynamicRangeIsSet() const TRTNOEXCEPT = 0;

    //!
    //! \brief Undo effect of setDynamicRange.
    //!
    virtual void resetDynamicRange() TRTNOEXCEPT = 0;

    //!
    //! \brief Get minimum of dynamic range.
    //!
    //! \return Minimum of dynamic range, or quiet NaN if range was not set.
    //!
    virtual float getDynamicRangeMin() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get maximum of dynamic range.
    //!
    //! \return Maximum of dynamic range, or quiet NaN if range was not set.
    //!
    virtual float getDynamicRangeMax() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set allowed formats.
    //!
    //! \param formats A bitmask of TensorFormat values that are supported for this tensor.
    //!
    //! \see ITensor::getAllowedFormats()
    //!
    virtual void setAllowedFormats(TensorFormats formats) TRTNOEXCEPT = 0;

    //!
    //! \brief Get a bitmask of TensorFormat values that the tensor supports.
    //!
    //! \see ITensor::getAllowReformat(), ITensor::setAllowedFormats()
    //!
    virtual TensorFormats getAllowedFormats() const TRTNOEXCEPT = 0;

    //!
    //! \brief Whether the tensor is a shape tensor.
    //!
    //! If a tensor is a shape tensor and becomes an engine input or output,
    //! then ICudaEngine::isShapeBinding will be true for that tensor.
    //!
    //! It is possible for a tensor to be both a shape tensor and an execution tensor.
    //!
    //! \return True if tensor is a shape tensor, false otherwise.
    //!
    virtual bool isShapeTensor() const TRTNOEXCEPT = 0;

    //!
    //! \brief Whether the tensor is an execution tensor.
    //!
    //! If a tensor is an execution tensor and becomes an engine input or output,
    //! then ICudaEngine::isExecutionBinding will be true for that tensor.
    //!
    //! Tensors are usually execution tensors.  The exceptions are tensors used
    //! solely for shape calculations or whose contents not needed to compute the outputs.
    //!
    //! A tensor with isShapeTensor() == false and isExecutionTensor() == false
    //! can still show up as an input to the engine if its dimensions are required.
    //! In that case, only its dimensions need to be set at runtime and a nullptr
    //! can be passed instead of a pointer to its contents.
    //!
    virtual bool isExecutionTensor() const TRTNOEXCEPT = 0;
};

//!
//! \class ILayer
//!
//! \brief Base class for all layer classes in a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ILayer
{
public:
    //!
    //! \brief Return the type of a layer.
    //!
    //! \see LayerType
    //!
    virtual LayerType getType() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the name of a layer.
    //!
    //! This method copies the name string.
    //!
    //! \see getName()
    //!
    virtual void setName(const char* name) TRTNOEXCEPT = 0;

    //!
    //! \brief Return the name of a layer.
    //!

    //! \see setName()
    //!
    virtual const char* getName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the number of inputs of a layer.
    //!
    virtual int getNbInputs() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the layer input corresponding to the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range or the tensor is optional
    //! (\ref IRNNLayer and \ref IRNNv2Layer).
    //!
    virtual ITensor* getInput(int index) const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the number of outputs of a layer.
    //!
    virtual int getNbOutputs() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the layer output corresponding to the given index.
    //!
    //! \return The indexed output tensor, or nullptr if the index is out of range or the tensor is optional
    //! (\ref IRNNLayer and \ref IRNNv2Layer).
    //!
    virtual ITensor* getOutput(int index) const TRTNOEXCEPT = 0;

    //!
    //! \brief replace an input of this layer with a specific tensor
    //!
    //! Except of IShuffleLayer and ISliceLayer, this method cannot change the number of inputs to a layer.
    //! The index argument must be less than the value of getNbInputs().
    //!
    //! See comments for IShuffleLayer::setInput() and ISliceLayer::setInput() for their special behavior.
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    virtual void setInput(int index, ITensor& tensor) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the computational precision of this layer
    //!
    //! setting the precision forces TensorRT to choose implementations which run at this precision. If precision is
    //! not set, TensorRT will select the computational precision based on performance considerations and the flags
    //! specified to the builder.
    //!
    //! \param precision the computational precision.
    //!
    //! \see getPrecision() precisionIsSet() resetPrecision()

    virtual void setPrecision(DataType dataType) TRTNOEXCEPT = 0;

    //!
    //! \brief get the computational precision of this layer
    //!
    //! \return the computational precision
    //!
    //! \see setPrecision() precisionIsSet() resetPrecision()

    virtual DataType getPrecision() const TRTNOEXCEPT = 0;

    //!
    //! \brief whether the computational precision has been set for this layer
    //!
    //! \return whether the computational precision has been explicitly set
    //!
    //! \see setPrecision() getPrecision() resetPrecision()

    virtual bool precisionIsSet() const TRTNOEXCEPT = 0;

    //!
    //! \brief reset the computational precision for this layer
    //!
    //! \see setPrecision() getPrecision() precisionIsSet()

    virtual void resetPrecision() TRTNOEXCEPT = 0;

    //!
    //! \brief Set the output type of this layer
    //!
    //! Setting the output type constrains TensorRT to choose implementations which generate output data with the
    //! given type. If it is not set, TensorRT will select the implementation based on performance considerations
    //! and the flags specified to the builder. Note that this method cannot be used to set the data type of the
    //! second output tensor of the topK layer. The data type of the second output tensor of the topK layer is
    //! always Int32.
    //!
    //! \param index the index of the output to set
    //! \param dataType the type of the output
    //!
    //! \see getOutputType() outputTypeIsSet() resetOutputType()

    virtual void setOutputType(int index, DataType dataType) TRTNOEXCEPT = 0;

    //!
    //! \brief get the output type of this layer
    //!
    //! \param index the index of the output
    //! \return the output precision. If no precision has been set, DataType::kFLOAT will be returned,
    //!         unless the output type is inherently DataType::kINT32.
    //!
    //! \see getOutputType() outputTypeIsSet() resetOutputType()

    virtual DataType getOutputType(int index) const TRTNOEXCEPT = 0;

    //!
    //! \brief whether the output type has been set for this layer
    //!
    //! \param index the index of the output
    //! \return whether the output type has been explicitly set
    //!
    //! \see setOutputType() getOutputType() resetOutputType()

    virtual bool outputTypeIsSet(int index) const TRTNOEXCEPT = 0;

    //!
    //! \brief reset the output type for this layer
    //!
    //! \param index the index of the output
    //!
    //! \see setOutputType() getOutputType() outputTypeIsSet()

    virtual void resetOutputType(int index) TRTNOEXCEPT = 0;

protected:
    virtual ~ILayer() {}
};

//!
//! \enum PaddingMode
//!
//! \brief Enumerates the modes of padding to perform in convolution, deconvolution and pooling layer,
//! padding mode takes precedence if setPaddingMode() and setPrePadding() are also used.
//!
//! kEXPLICIT* padding is to use explicit padding.
//! kSAME* padding is to implicitly calculate padding to keep output dim to be the "same" with input dim. For
//! convolution and pooling, output dim is ceil(input dim, stride), for deconvolution it is inverse, then use
//! the output dim to calculate padding size. kCAFFE* padding is symmetric padding.
//!
enum class PaddingMode : int
{
    kEXPLICIT_ROUND_DOWN = 0, //!< Use explicit padding, rounding output size down.
    kEXPLICIT_ROUND_UP = 1,   //!< Use explicit padding, rounding output size up.
    kSAME_UPPER = 2,          //!< Use SAME padding with prePadding <= postPadding.
    kSAME_LOWER = 3,          //!< Use SAME padding, with prePadding >= postPadding.
    kCAFFE_ROUND_DOWN = 4,    //!< Use CAFFE padding, rounding output size down.
    kCAFFE_ROUND_UP = 5       //!< Use CAFFE padding, rounding output size up.
};

template <>
constexpr inline int EnumMax<PaddingMode>()
{
    return 6;
} //!< Maximum number of elements in PaddingMode enum. \see PaddingMode

//!
//! \class IConvolutionLayer
//!
//! \brief A convolution layer in a network definition.
//!
//! This layer performs a correlation operation between 3-dimensional filter with a 4-dimensional tensor to produce
//! another 4-dimensional tensor.
//!
//! The HW output size of the convolution is set according to the \p INetworkCustomDimensions set in
//! INetworkDefinition::setCustomConvolutionDimensions().
//!
//! An optional bias argument is supported, which adds a per-channel constant to each value in the output.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConvolutionLayer : public ILayer
{
public:
    //!
    //! \brief Set the HW kernel size of the convolution.
    //!
    //! If executing this layer on DLA, both height and width of kernel size must be in the range [1,16].
    //!
    //! \see getKernelSize()
    //!
    virtual void setKernelSize(DimsHW kernelSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the HW kernel size of the convolution.
    //!
    //! \see setKernelSize()
    //!
    virtual DimsHW getKernelSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the number of output maps for the convolution.
    //!
    //! If executing this layer on DLA, the number of output maps must be in the range [1,8192].
    //!
    //! \see getNbOutputMaps()
    //!
    virtual void setNbOutputMaps(int nbOutputMaps) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the number of output maps for the convolution.
    //!
    //! \see setNbOutputMaps()
    //!
    virtual int getNbOutputMaps() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the stride of the convolution.
    //!
    //! Default: (1,1)
    //!
    //! If executing this layer on DLA, both height and width of stride must be in the range [1,8].
    //!
    //! \see getStride()
    //!
    virtual void setStride(DimsHW stride) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the stride of the convolution.
    //!
    virtual DimsHW getStride() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the padding of the convolution.
    //!
    //! The input will be zero-padded by this number of elements in the height and width directions.
    //! Padding is symmetric.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPadding(DimsHW padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the padding of the convolution. If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPadding()
    //!
    virtual DimsHW getPadding() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the number of groups for a convolution.
    //!
    //! The input tensor channels are  divided into \p nbGroups groups, and a convolution is executed for each group,
    //! using a filter per group. The results of the group convolutions are concatenated to form the output.
    //!
    //! \note When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group
    //! count) must be a multiple of 4 for both input and output.
    //!
    //! Default: 1
    //!
    //! \see getNbGroups()
    //!
    virtual void setNbGroups(int nbGroups) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the number of groups for a convolution.
    //!
    //! \see setNbGroups()
    //!
    virtual int getNbGroups() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the kernel weights for the convolution.
    //!
    //! The weights are specified as a contiguous array in \p GKCRS order, where \p G is the number of groups, \p K
    //! the number of output feature maps, \p C the number of input channels, and \p R and \p S are the height and
    //! width of the filter.
    //!
    //! \see getKernelWeights()
    //!
    virtual void setKernelWeights(Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the kernel weights for the convolution.
    //!
    //! \see setKernelWeights()
    //!
    virtual Weights getKernelWeights() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the bias weights for the convolution.
    //!
    //! Bias is optional. To omit bias, set the count value of the weights structure to zero.
    //!
    //! The bias is applied per-channel, so the number of weights (if non-zero) must be equal to the number of output
    //! feature maps.
    //!
    //! \see getBiasWeights()
    //!
    virtual void setBiasWeights(Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the bias weights for the convolution.
    //!
    //! \see setBiasWeights()
    //!
    virtual Weights getBiasWeights() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the dilation for a convolution.
    //!
    //! Default: (1,1)
    //!
    //! \see getDilation()
    //!
    virtual void setDilation(DimsHW dilation) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the dilation for a convolution.
    //!
    //! \see setDilation()
    //!
    virtual DimsHW getDilation() const TRTNOEXCEPT = 0;

protected:
    virtual ~IConvolutionLayer() {}

public:
    //!
    //! \brief Set the pre-padding.
    //!
    //! The start of input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: 0
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPrePadding()
    //!
    virtual void setPrePadding(Dims padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    virtual Dims getPrePadding() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the post-padding.
    //!
    //! The end of the input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPostPadding()
    //!
    virtual void setPostPadding(Dims padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the post-padding.
    //!
    //! \see setPostPadding()
    //!
    virtual Dims getPostPadding() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode takes precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    //!
    virtual void setPaddingMode(PaddingMode paddingMode) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    //!
    virtual PaddingMode getPaddingMode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension kernel size of the convolution.
    //!
    //! If executing this layer on DLA, only support 2D kernel size, both height and width of kernel size must be in the range [1,16].
    //!
    //! \see getKernelSizeNd() setKernelSize() getKernelSize()
    //!
    virtual void setKernelSizeNd(Dims kernelSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension kernel size of the convolution.
    //!
    //! \see setKernelSizeNd()
    //!
    virtual Dims getKernelSizeNd() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension stride of the convolution.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! If executing this layer on DLA, only support 2D stride, both height and width of stride must be in the range [1,8].
    //!
    //! \see getStrideNd() setStride() getStride()
    //!
    virtual void setStrideNd(Dims stride) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension stride of the convolution.
    //!
    //! \see setStrideNd()
    //!
    virtual Dims getStrideNd() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension padding of the convolution.
    //!
    //! The input will be zero-padded by this number of elements in each dimension.
    //! Padding is symmetric.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPaddingNd() setPadding() getPadding()
    //!
    virtual void setPaddingNd(Dims padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension padding of the convolution.
    //!
    //! If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPaddingNd()
    //!
    virtual Dims getPaddingNd() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension dilation of the convolution.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! \see getDilation()
    //!
    virtual void setDilationNd(Dims dilation) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension dilation of the convolution.
    //!
    //! \see setDilation()
    //!
    virtual Dims getDilationNd() const TRTNOEXCEPT = 0;
};

//! \class IFullyConnectedLayer
//!
//! \brief A fully connected layer in a network definition.
//! This layer expects an input tensor of three or more non-batch dimensions.  The input is automatically
//! reshaped into an `MxV` tensor `X`, where `V` is a product of the last three dimensions and `M`
//! is a product of the remaining dimensions (where the product over 0 dimensions is defined as 1).  For example:
//!
//! - If the input tensor has shape `{C, H, W}`, then the tensor is reshaped into `{1, C*H*W}`.
//! - If the input tensor has shape `{P, C, H, W}`, then the tensor is reshaped into `{P, C*H*W}`.
//!
//! The layer then performs the following operation:
//!
//! ~~~
//! Y := matmul(X, W^T) + bias
//! ~~~
//!
//! Where `X` is the `MxV` tensor defined above, `W` is the `KxV` weight tensor
//! of the layer, and `bias` is a row vector size `K` that is broadcasted to
//! `MxK`.  `K` is the number of output channels, and configurable via
//! setNbOutputChannels().  If `bias` is not specified, it is implicitly `0`.
//!
//! The `MxK` result `Y` is then reshaped such that the last three dimensions are `{K, 1, 1}` and
//! the remaining dimensions match the dimensions of the input tensor. For example:
//!
//! - If the input tensor has shape `{C, H, W}`, then the output tensor will have shape `{K, 1, 1}`.
//! - If the input tensor has shape `{P, C, H, W}`, then the output tensor will have shape `{P, K, 1, 1}`.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IFullyConnectedLayer : public ILayer
{
public:
    //!
    //! \brief Set the number of output channels `K` from the fully connected layer.
    //!
    //! If executing this layer on DLA, number of output channels must in the range [1,8192].
    //!
    //! \see getNbOutputChannels()
    //!
    virtual void setNbOutputChannels(int nbOutputs) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the number of output channels `K` from the fully connected layer.
    //!
    //! \see setNbOutputChannels()
    //!
    virtual int getNbOutputChannels() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the kernel weights, given as a `KxC` matrix in row-major order.
    //!
    //! \see getKernelWeights()
    //!
    virtual void setKernelWeights(Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the kernel weights.
    //!
    //! \see setKernelWeights()
    //!
    virtual Weights getKernelWeights() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the bias weights.
    //!
    //! Bias is optional. To omit bias, set the count value in the weights structure to zero.
    //!
    //! \see getBiasWeightsWeights()
    //!
    virtual void setBiasWeights(Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the bias weights.
    //!
    //! \see setBiasWeightsWeights()
    //!
    virtual Weights getBiasWeights() const TRTNOEXCEPT = 0;

protected:
    virtual ~IFullyConnectedLayer() {}
};

template <>
constexpr inline int EnumMax<ActivationType>()
{
    return 12;
} //!< Maximum number of elements in ActivationType enum. \see ActivationType

//!
//! \class IActivationLayer
//!
//! \brief An Activation layer in a network definition.
//!
//! This layer applies a per-element activation function to its input.
//!
//! The output has the same shape as the input.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IActivationLayer : public ILayer
{
public:
    //!
    //! \brief Set the type of activation to be performed.
    //!
    //! \see getActivationType(), ActivationType
    //!
    virtual void setActivationType(ActivationType type) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the type of activation to be performed.
    //!
    //! \see setActivationType(), ActivationType
    //!
    virtual ActivationType getActivationType() const TRTNOEXCEPT = 0;

protected:
    virtual ~IActivationLayer() {}
public:
    //!
    //! \brief Set the alpha parameter (must be finite).
    //!
    //! This parameter is used by the following activations:
    //! LeakyRelu, Elu, Selu, Softplus, Clip, HardSigmoid, ScaledTanh,
    //! ThresholdedRelu.
    //!
    //! It is ignored by the other activations.
    //!
    //! \see getAlpha(), setBeta()
    virtual void setAlpha(float alpha) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the beta parameter (must be finite).
    //!
    //! This parameter is used by the following activations:
    //! Selu, Softplus, Clip, HardSigmoid, ScaledTanh.
    //!
    //! It is ignored by the other activations.
    //!
    //! \see getBeta(), setAlpha()
    virtual void setBeta(float beta) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the alpha parameter.
    //!
    //! \see getBeta(), setAlpha()
    virtual float getAlpha() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the beta parameter.
    //!
    //! \see getAlpha(), setBeta()
    virtual float getBeta() const TRTNOEXCEPT = 0;
};

//!
//! \enum PoolingType
//!
//! \brief The type of pooling to perform in a pooling layer.
//!
enum class PoolingType : int
{
    kMAX = 0,              // Maximum over elements
    kAVERAGE = 1,          // Average over elements. If the tensor is padded, the count includes the padding
    kMAX_AVERAGE_BLEND = 2 // Blending between max and average pooling: (1-blendFactor)*maxPool + blendFactor*avgPool
};

template <>
constexpr inline int EnumMax<PoolingType>()
{
    return 3;
} //!< Maximum number of elements in PoolingType enum. \see PoolingType

//! \class IPoolingLayer
//!
//! \brief A Pooling layer in a network definition.
//!
//! The layer applies a reduction operation within a window over the input.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPoolingLayer : public ILayer
{
public:
    //!
    //! \brief Set the type of activation to be performed.
    //!
    //! DLA only supports kMAX and kAVERAGE.
    //!
    //! \see getPoolingType(), PoolingType
    //!
    virtual void setPoolingType(PoolingType type) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the type of activation to be performed.
    //!
    //! \see setPoolingType(), PoolingType
    //!
    virtual PoolingType getPoolingType() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the window size for pooling.
    //!
    //! If executing this layer on DLA, both height and width of window size must be in the range [1,8].
    //!
    //! \see getWindowSize()
    //!
    virtual void setWindowSize(DimsHW windowSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the window size for pooling.
    //!
    //! \see setWindowSize()
    //!
    virtual DimsHW getWindowSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the stride for pooling.
    //!
    //! Default: 1
    //!
    //! If executing this layer on DLA, both height and width of stride must be in the range [1,16].
    //!
    //! \see getStride()
    //!
    virtual void setStride(DimsHW stride) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the stride for pooling.
    //!
    //! \see setStride()
    //!
    virtual DimsHW getStride() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the padding for pooling.
    //!
    //! Default: 0
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,7].
    //!
    //! \see getPadding()
    //!
    virtual void setPadding(DimsHW padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the padding for pooling.
    //!
    //! Default: 0
    //!
    //! \see setPadding()
    //!
    virtual DimsHW getPadding() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the blending factor for the max_average_blend mode:
    //! max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
    //! blendFactor is a user value in [0,1] with the default value of 0.0
    //! This value only applies for the kMAX_AVERAGE_BLEND mode.
    //!
    //! \see getBlendFactor()
    //!
    virtual void setBlendFactor(float blendFactor) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the blending factor for the max_average_blend mode:
    //! max_average_blendPool = (1-blendFactor)*maxPool + blendFactor*avgPool
    //! blendFactor is a user value in [0,1] with the default value of 0.0
    //! In modes other than kMAX_AVERAGE_BLEND, blendFactor is ignored.
    //!
    //! \see setBlendFactor()
    //!
    virtual float getBlendFactor() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set whether average pooling uses as a denominator the overlap area between the window
    //! and the unpadded input.
    //! If this is not set, the denominator is the overlap between the pooling window and the padded input.
    //!
    //! Default: true
    //!
    //! \see getAverageCountExcludesPadding()
    //!
    virtual void setAverageCountExcludesPadding(bool exclusive) TRTNOEXCEPT = 0;

    //!
    //! \brief Get whether exclusive pooling uses as a denominator the overlap area betwen the window
    //! and the unpadded input.
    //!
    //! \see setAverageCountExcludesPadding()
    //!
    virtual bool getAverageCountExcludesPadding() const TRTNOEXCEPT = 0;

protected:
    virtual ~IPoolingLayer() {}

public:
    //!
    //! \brief Set the pre-padding.
    //!
    //! The start of input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: 0
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPrePadding(Dims padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    virtual Dims getPrePadding() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the post-padding.
    //!
    //! The end of the input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPostPadding(Dims padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the padding.
    //!
    //! \see setPadding()
    //!
    virtual Dims getPostPadding() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode takes precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    virtual void setPaddingMode(PaddingMode paddingMode) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    virtual PaddingMode getPaddingMode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension window size for pooling.
    //!
    //! If executing this layer on DLA, only support 2D window size, both height and width of window size must be in the range [1,8].
    //!
    //! \see getWindowSizeNd() setWindowSize() getWindowSize()
    //!
    virtual void setWindowSizeNd(Dims windowSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension window size for pooling.
    //!
    //! \see setWindowSizeNd()
    //!
    virtual Dims getWindowSizeNd() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension stride for pooling.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! If executing this layer on DLA, only support 2D stride, both height and width of stride must be in the range [1,16].
    //!
    //! \see getStrideNd() setStride() getStride()
    //!
    virtual void setStrideNd(Dims stride) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension stride for pooling.
    //!
    //! \see setStrideNd()
    //!
    virtual Dims getStrideNd() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension padding for pooling.
    //!
    //! The input will be zero-padded by this number of elements in each dimension.
    //! Padding is symmetric.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width of padding must be in the range [0,7].
    //!
    //! \see getPaddingNd() setPadding() getPadding()
    //!
    virtual void setPaddingNd(Dims padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension padding for pooling.
    //!
    //! If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPaddingNd()
    //!
    virtual Dims getPaddingNd() const TRTNOEXCEPT = 0;
};

//!
//! \class ILRNLayer
//!
//! \brief A LRN layer in a network definition.
//!
//! The output size is the same as the input size.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ILRNLayer : public ILayer
{
public:
    //!
    //! \brief Set the LRN window size.
    //!
    //! The window size must be odd and in the range of [1, 15].
    //! \see setWindowStride()
    //!
    virtual void setWindowSize(int windowSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the LRN window size.
    //!
    //! \see getWindowStride()
    //!
    virtual int getWindowSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the LRN alpha value.
    //!
    //! The valid range is [-1e20, 1e20].
    //! \see getAlpha()
    //!
    virtual void setAlpha(float alpha) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the LRN alpha value.
    //!
    //! \see setAlpha()
    //!
    virtual float getAlpha() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the LRN beta value.
    //!
    //! The valid range is [0.01, 1e5f].
    //! \see getBeta()
    //!
    virtual void setBeta(float beta) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the LRN beta value.
    //!
    //! \see setBeta()
    //!
    virtual float getBeta() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the LRN K value.
    //!
    //! The valid range is [1e-5, 1e10].
    //! \see getK()
    //!
    virtual void setK(float k) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the LRN K value.
    //!
    //! \see setK()
    //!
    virtual float getK() const TRTNOEXCEPT = 0;

protected:
    virtual ~ILRNLayer() {}
};

//!
//! \brief Controls how shift, scale and power are applied in a Scale layer.
//!
//! \see IScaleLayer
//!
enum class ScaleMode : int
{
    kUNIFORM = 0,    //!< Identical coefficients across all elements of the tensor.
    kCHANNEL = 1,    //!< Per-channel coefficients. The channel dimension is assumed to be the third to last dimension
    kELEMENTWISE = 2 //!< Elementwise coefficients.
};

template <>
constexpr inline int EnumMax<ScaleMode>()
{
    return 3;
} //!< Maximum number of elements in ScaleMode enum. \see ScaleMode

//!
//! \class IScaleLayer
//!
//! \brief A Scale layer in a network definition.
//!
//! This layer applies a per-element computation to its input:
//!
//! \p output = (\p input* \p scale + \p shift)^ \p power
//!
//! The coefficients can be applied on a per-tensor, per-channel, or per-element basis.
//!
//! \note If the number of weights is 0, then a default value is used for shift, power, and scale.
//!       The default shift is 0, the default power is 1, and the default scale is 1.
//!
//! The output size is the same as the input size.
//!
//! \note The input tensor for this layer is required to have a minimum of 3 dimensions.
//!
//! \see ScaleMode
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IScaleLayer : public ILayer
{
public:
    //!
    //! \brief Set the scale mode.
    //!
    //! \see getMode()
    //!
    virtual void setMode(ScaleMode mode) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the scale mode.
    //!
    //! \see setMode()
    //!
    virtual ScaleMode getMode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the shift value.
    //!
    //! \see getShift()
    //!
    virtual void setShift(Weights shift) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the shift value.
    //!
    //! \see setShift()
    //!
    virtual Weights getShift() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the scale value.
    //!
    //! \see getScale()
    //!
    virtual void setScale(Weights scale) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the scale value.
    //!
    //! \see setScale()
    //!
    virtual Weights getScale() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the power value.
    //!
    //! \see getPower()
    //!
    virtual void setPower(Weights power) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the power value.
    //!
    //! \see setPower()
    //!
    virtual Weights getPower() const TRTNOEXCEPT = 0;

protected:
    virtual ~IScaleLayer() {}
};

//!
//! \class ISoftMaxLayer
//!
//! \brief A Softmax layer in a network definition.
//!
//! This layer applies a per-channel softmax to its input.
//!
//! The output size is the same as the input size.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ISoftMaxLayer : public ILayer
{
protected:
    virtual ~ISoftMaxLayer() {}
public:
    //!
    //! \brief Set the axis along which softmax is computed. Currently, only one axis can be set.
    //!
    //! The axis is specified by setting the bit corresponding to the axis, after excluding the batch dimension, to 1.
    //! Let's say we have an NCHW tensor as input (three non-batch dimensions).
    //! Bit 0 corresponds to the C dimension boolean.
    //! Bit 1 corresponds to the H dimension boolean.
    //! Bit 2 corresponds to the W dimension boolean.
    //! For example, to perform softmax on axis R of a NPQRCHW input, set bit 2.
    //!
    //! By default, softmax is performed on the axis which is the number of non-batch axes minus three. It is 0 if
    //! there are fewer than 3 non-batch axes. For example, if the input is NCHW, the default axis is C. If the input
    //! is NHW, then the default axis is H.
    //!
    //! \param axes The axis along which softmax is computed.
    //!
    virtual void setAxes(uint32_t axes) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the axis along which softmax occurs.
    //!
    //! \see setAxes()
    //!
    virtual uint32_t getAxes() const TRTNOEXCEPT = 0;
};

//!
//! \class IConcatenationLayer
//!
//! \brief A concatenation layer in a network definition.
//!
//! The output channel size is the sum of the channel sizes of the inputs.
//! The other output sizes are the same as the other input sizes,
//! which must all match.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConcatenationLayer : public ILayer
{
protected:
    virtual ~IConcatenationLayer() {}

public:
    //!
    //! \brief Set the axis along which concatenation occurs.
    //!
    //! 0 is the major axis (excluding the batch dimension). The default is the number of non-batch axes in the tensor
    //! minus three (e.g. for an NCHW input it would be 0), or 0 if there are fewer than 3 non-batch axes.
    //!
    //! \param axis The axis along which concatenation occurs.
    //!
    virtual void setAxis(int axis) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the axis along which concatenation occurs.
    //!
    //! \see setAxis()
    //!
    virtual int getAxis() const TRTNOEXCEPT = 0;
};

//!
//! \class IDeconvolutionLayer
//!
//! \brief A deconvolution layer in a network definition.
//!
//! The output size is defined using the formula set by INetworkDefinition::setDeconvolutionOutputDimensionsFormula().
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IDeconvolutionLayer : public ILayer
{
public:
    //!
    //! \brief Set the HW kernel size of the convolution.
    //!
    //! If executing this layer on DLA, both height and width of kernel size must be in the range [1,16].
    //!
    //! \see getKernelSize()
    //!
    virtual void setKernelSize(DimsHW kernelSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the HW kernel size of the deconvolution.
    //!
    //! \see setKernelSize()
    //!
    virtual DimsHW getKernelSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the number of output feature maps for the deconvolution.
    //!
    //! If executing this layer on DLA, the number of output maps must be in the range [1,8192].
    //!
    //! \see getNbOutputMaps()
    //!
    virtual void setNbOutputMaps(int nbOutputMaps) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the number of output feature maps for the deconvolution.
    //!
    //! \see setNbOutputMaps()
    //!
    virtual int getNbOutputMaps() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the stride of the deconvolution.
    //!
    //! If executing this layer on DLA, both height and width of stride must be in the range [1,8].
    //!
    //! \see setStride()
    //!
    virtual void setStride(DimsHW stride) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the stride of the deconvolution.
    //!
    //! Default: (1,1)
    //!
    virtual DimsHW getStride() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the padding of the deconvolution.
    //!
    //! The output will be trimmed by this number of elements on each side in the height and width directions.
    //! In other words, it resembles the inverse of a convolution layer with this padding size.
    //! Padding is symmetric, and negative padding is not supported.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPadding(DimsHW padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the padding of the deconvolution.
    //!
    //! \see setPadding()
    //!
    virtual DimsHW getPadding() const TRTNOEXCEPT = 0; // padding defaults to 0

    //!
    //! \brief Set the number of groups for a deconvolution.
    //!
    //! The input tensor channels are divided into \p nbGroups groups, and a deconvolution is executed for each group,
    //! using a filter per group. The results of the group convolutions are concatenated to form the output.
    //!
    //! \note When using groups in int8 mode, the size of the groups (i.e. the channel count divided by the group count)
    //! must be a multiple of 4 for both input and output.
    //!
    //! Default: 1
    //!
    //! \see getNbGroups()
    //!
    virtual void setNbGroups(int nbGroups) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the number of groups for a deconvolution.
    //!
    //! \see setNbGroups()
    //!
    virtual int getNbGroups() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the kernel weights for the deconvolution.
    //!
    //! The weights are specified as a contiguous array in \p CKRS order, where \p C the number of
    //! input channels, \p K the number of output feature maps, and \p R and \p S are the height and width
    //! of the filter.
    //!
    //! \see getWeights()
    //!
    virtual void setKernelWeights(Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the kernel weights for the deconvolution.
    //!
    //! \see setNbGroups()
    //!
    virtual Weights getKernelWeights() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the bias weights for the deconvolution.
    //!
    //! Bias is optional. To omit bias, set the count value of the weights structure to zero.
    //!
    //! The bias is applied per-feature-map, so the number of weights (if non-zero) must be equal to the number of
    //! output feature maps.
    //!
    //! \see getBiasWeights()
    //!
    virtual void setBiasWeights(Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the bias weights for the deconvolution.
    //!
    //! \see getBiasWeights()
    //!
    virtual Weights getBiasWeights() const TRTNOEXCEPT = 0;

protected:
    virtual ~IDeconvolutionLayer() {}

public:
    //!
    //! \brief Set the pre-padding.
    //!
    //! The start of input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: 0
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPrePadding(Dims padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the pre-padding.
    //!
    //! \see setPrePadding()
    //!
    virtual Dims getPrePadding() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the post-padding.
    //!
    //! The end of the input will be zero-padded by this number of elements in the height and width directions.
    //!
    //! Default: (0,0)
    //!
    //! If executing this layer on DLA, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPadding()
    //!
    virtual void setPostPadding(Dims padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the padding.
    //!
    //! \see setPadding()
    //!
    virtual Dims getPostPadding() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the padding mode.
    //!
    //! Padding mode takes precedence if both setPaddingMode and setPre/PostPadding are used.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see getPaddingMode()
    virtual void setPaddingMode(PaddingMode paddingMode) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the padding mode.
    //!
    //! Default: kEXPLICIT_ROUND_DOWN
    //!
    //! \see setPaddingMode()
    virtual PaddingMode getPaddingMode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension kernel size of the deconvolution.
    //!
    //! If executing this layer on DLA, only support 2D kernel size, both height and width of kernel size must be in the range [1,16].
    //!
    //! \see getKernelSizeNd() setKernelSize() getKernelSize()
    //!
    virtual void setKernelSizeNd(Dims kernelSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension kernel size of the deconvolution.
    //!
    //! \see setKernelSizeNd()
    //!
    virtual Dims getKernelSizeNd() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension stride of the deconvolution.
    //!
    //! Default: (1, 1, ..., 1)
    //!
    //! If executing this layer on DLA, only support 2D stride, both height and width of stride must be in the range [1,8].
    //!
    //! \see getStrideNd() setStride() getStride()
    //!
    virtual void setStrideNd(Dims stride) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension stride of the deconvolution.
    //!
    //! \see setStrideNd()
    //!
    virtual Dims getStrideNd() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the multi-dimension padding of the deconvolution.
    //!
    //! The input will be zero-padded by this number of elements in each dimension.
    //! Padding is symmetric.
    //!
    //! Default: (0, 0, ..., 0)
    //!
    //! If executing this layer on DLA, only support 2D padding, both height and width of padding must be in the range [0,15].
    //!
    //! \see getPaddingNd() setPadding() getPadding()
    //!
    virtual void setPaddingNd(Dims padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the multi-dimension padding of the deconvolution.
    //!
    //! If the padding is asymmetric, the pre-padding is returned.
    //!
    //! \see setPaddingNd()
    //!
    virtual Dims getPaddingNd() const TRTNOEXCEPT = 0;
};

//!
//! \enum ElementWiseOperation
//!
//! \brief Enumerates the binary operations that may be performed by an ElementWise layer.
//!
//! \see IElementWiseLayer
//!
enum class ElementWiseOperation : int
{
    kSUM = 0,      //!< Sum of the two elements.
    kPROD = 1,     //!< Product of the two elements.
    kMAX = 2,      //!< Maximum of the two elements.
    kMIN = 3,      //!< Minimum of the two elements.
    kSUB = 4,      //!< Substract the second element from the first.
    kDIV = 5,      //!< Divide the first element by the second.
    kPOW = 6,      //!< The first element to the power of the second element.
    kFLOOR_DIV = 7 //!< Floor division of the first element by the second.
};

template <>
constexpr inline int EnumMax<ElementWiseOperation>()
{
    return 8;
} //!< Maximum number of elements in ElementWiseOperation enum. \see ElementWiseOperation

//!
//! \class IElementWiseLayer
//!
//! \brief A elementwise layer in a network definition.
//!
//! This layer applies a per-element binary operation between corresponding elements of two tensors.
//!
//! The input dimensions of the two input tensors must be equal, and the output tensor is the same size as each input.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IElementWiseLayer : public ILayer
{
public:
    //!
    //! \brief Set the binary operation for the layer.
    //!
    //! DLA supports only kSUM, kPROD, kMAX and kMIN.
    //!
    //! \see getOperation(), ElementWiseOperation
    //!
    //! \see getBiasWeights()
    //!
    virtual void setOperation(ElementWiseOperation type) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the binary operation for the layer.
    //!
    //! \see setOperation(), ElementWiseOperation
    //!
    //! \see setBiasWeights()
    //!
    virtual ElementWiseOperation getOperation() const TRTNOEXCEPT = 0;

protected:
    virtual ~IElementWiseLayer() {}
};

//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IGatherLayer : public ILayer
{
public:
    //!
    //! \brief Set the non-batch dimension axis to gather on.
    //!  The axis must be less than the number of non-batch dimensions in the data input.
    //!
    //! \see getGatherAxis()
    //!
    virtual void setGatherAxis(int axis) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the non-batch dimension axis to gather on.
    //!
    //! \see setGatherAxis()
    //!
    virtual int getGatherAxis() const TRTNOEXCEPT = 0;

protected:
    virtual ~IGatherLayer() {}
};

//!
//! \enum RNNOperation
//!
//! \brief Enumerates the RNN operations that may be performed by an RNN layer.
//!
//! __Equation definitions__
//!
//! In the equations below, we use the following naming convention:
//!
//! ~~~
//! t := current time step
//!
//! i := input gate
//! o := output gate
//! f := forget gate
//! z := update gate
//! r := reset gate
//! c := cell gate
//! h := hidden gate
//!
//! g[t] denotes the output of gate g at timestep t, e.g.
//! f[t] is the output of the forget gate f.
//!
//! X[t] := input tensor for timestep t
//! C[t] := cell state for timestep t
//! H[t] := hidden state for timestep t
//!
//! W[g] := W (input) parameter weight matrix for gate g
//! R[g] := U (recurrent) parameter weight matrix for gate g
//! Wb[g] := W (input) parameter bias vector for gate g
//! Rb[g] := U (recurrent) parameter bias vector for gate g
//!
//! Unless otherwise specified, all operations apply pointwise
//! to elements of each operand tensor.
//!
//! ReLU(X) := max(X, 0)
//! tanh(X) := hyperbolic tangent of X
//! sigmoid(X) := 1 / (1 + exp(-X))
//! exp(X) := e^X
//!
//! A.B denotes matrix multiplication of A and B.
//! A*B denotes pointwise multiplication of A and B.
//! ~~~
//!
//! __Equations__
//!
//! Depending on the value of RNNOperation chosen, each sub-layer of the RNN
//! layer will perform one of the following operations:
//!
//! ~~~
//! ::kRELU
//!
//!   H[t] := ReLU(W[i].X[t] + R[i].H[t-1] + Wb[i] + Rb[i])
//!
//! ::kTANH
//!
//!   H[t] := tanh(W[i].X[t] + R[i].H[t-1] + Wb[i] + Rb[i])
//!
//! ::kLSTM
//!
//!   i[t] := sigmoid(W[i].X[t] + R[i].H[t-1] + Wb[i] + Rb[i])
//!   f[t] := sigmoid(W[f].X[t] + R[f].H[t-1] + Wb[f] + Rb[f])
//!   o[t] := sigmoid(W[o].X[t] + R[o].H[t-1] + Wb[o] + Rb[o])
//!   c[t] :=    tanh(W[c].X[t] + R[c].H[t-1] + Wb[c] + Rb[c])
//!
//!   C[t] := f[t]*C[t-1] + i[t]*c[t]
//!   H[t] := o[t]*tanh(C[t])
//!
//! ::kGRU
//!
//!   z[t] := sigmoid(W[z].X[t] + R[z].H[t-1] + Wb[z] + Rb[z])
//!   r[t] := sigmoid(W[r].X[t] + R[r].H[t-1] + Wb[r] + Rb[r])
//!   h[t] := tanh(W[h].X[t] + r[t]*(R[h].H[t-1] + Rb[h]) + Wb[h])
//!
//!   H[t] := (1 - z[t])*h[t] + z[t]*H[t-1]
//! ~~~
//!
//! \see IRNNLayer, IRNNv2Layer
//!
enum class RNNOperation : int
{
    kRELU = 0, //!< Single gate RNN w/ ReLU activation function.
    kTANH = 1, //!< Single gate RNN w/ TANH activation function.
    kLSTM = 2, //!< Four-gate LSTM network w/o peephole connections.
    kGRU = 3   //!< Three-gate network consisting of Gated Recurrent Units.
};

template <>
constexpr inline int EnumMax<RNNOperation>()
{
    return 4;
} //!< Maximum number of elements in RNNOperation enum. \see RNNOperation

//!
//! \enum RNNDirection
//!
//! \brief Enumerates the RNN direction that may be performed by an RNN layer.
//!
//! \see IRNNLayer, IRNNv2Layer
//!
enum class RNNDirection : int
{
    kUNIDIRECTION = 0, //!< Network iterations from first input to last input.
    kBIDIRECTION = 1   //!< Network iterates from first to last and vice versa and outputs concatenated.
};

template <>
constexpr inline int EnumMax<RNNDirection>()
{
    return 2;
} //!< Maximum number of elements in RNNDirection enum. \see RNNDirection

//!
//! \enum RNNInputMode
//!
//! \brief Enumerates the RNN input modes that may occur with an RNN layer.
//!
//! If the RNN is configured with RNNInputMode::kLINEAR, then for each gate `g` in the first layer of the RNN,
//! the input vector `X[t]` (length `E`) is left-multiplied by the gate's corresponding weight matrix `W[g]`
//! (dimensions `HxE`) as usual, before being used to compute the gate output as described by \ref RNNOperation.
//!
//! If the RNN is configured with RNNInputMode::kSKIP, then this initial matrix multiplication is "skipped"
//! and `W[g]` is conceptually an identity matrix.  In this case, the input vector `X[t]` must have length `H`
//! (the size of the hidden state).
//!
//! \see IRNNLayer, IRNNv2Layer
//!
enum class RNNInputMode : int
{
    kLINEAR = 0, //!< Perform the normal matrix multiplication in the first recurrent layer.
    kSKIP = 1    //!< No operation is performed on the first recurrent layer.
};

template <>
constexpr inline int EnumMax<RNNInputMode>()
{
    return 2;
} //!< Maximum number of elements in RNNInputMode enum. \see RNNInputMode

//!
//! \class IRNNLayer
//!
//! \brief A RNN layer in a network definition.
//!
//! This layer applies an RNN operation on the inputs.
//!
//! \deprecated This interface is superseded by IRNNv2Layer.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class TRT_DEPRECATED IRNNLayer : public ILayer
{
public:
    //!
    //! \brief Get the number of layers in the RNN.
    //!
    //! \return The number of layers in the RNN.
    //!
    virtual unsigned getLayerCount() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the size of the hidden layers.
    //!
    //! The hidden size is the value of hiddenSize parameter passed into addRNN().
    //!
    //! \return The internal hidden layer size for the RNN.
    //! \see getDirection(), addRNN()
    //!
    virtual std::size_t getHiddenSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the sequence length.
    //!
    //! The sequence length is the maximum number of time steps passed into the addRNN() function.
    //! This is also the maximum number of input tensors that the RNN can process at once.
    //!
    //! \return the maximum number of time steps that can be executed by a single call RNN layer.
    //!
    virtual int getSeqLength() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the operation of the RNN layer.
    //!
    //! \see getOperation(), RNNOperation
    //!
    virtual void setOperation(RNNOperation op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the operation of the RNN layer.
    //!
    //! \see setOperation(), RNNOperation
    //!
    virtual RNNOperation getOperation() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the operation of the RNN layer.
    //!
    //! \see getInputMode(), RNNInputMode
    //!
    virtual void setInputMode(RNNInputMode op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the operation of the RNN layer.
    //!
    //! \see setInputMode(), RNNInputMode
    //!
    virtual RNNInputMode getInputMode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the direction of the RNN layer.
    //!
    //! The direction determines if the RNN is run
    //! as a unidirectional(left to right) or
    //! bidirectional(left to right and right to left).
    //! In the ::kBIDIRECTION case the
    //! output is concatenated together, resulting
    //! in output size of 2x getHiddenSize().
    //! \see getDirection(), RNNDirection
    //!
    virtual void setDirection(RNNDirection op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the direction of the RNN layer.
    //!
    //! \see setDirection(), RNNDirection
    //!
    virtual RNNDirection getDirection() const TRTNOEXCEPT = 0;

    //!
    //! \param weights The weight structure holding the weight parameters.
    //!
    //! \brief Set the weight parameters for the RNN.
    //!
    //! The trained weights for the weight parameter matrices of the RNN.
    //! The #DataType for this structure must be ::kFLOAT or ::kHALF, and must be the same
    //! datatype as the input tensor.
    //!
    //! The layout of the weight structure depends on the #RNNOperation, #RNNInputMode, and
    //! #RNNDirection of the layer.  The array specified by `weights.values` contains a sequence of
    //! parameter matrices, where each parameter matrix is linearly appended after the previous
    //! without padding; e.g., if parameter matrix 0 and 1 have M and N elements respectively, then
    //! the layout of `weights.values` in memory looks like:
    //!
    //! ~~~
    //! index | 0 1 2 3 4 ...  M-2 M-1 | M M+1  ... M+N-2 M+N-1 | M+N M+N+1 M+N+2 ...    | ...
    //! data  |-- parameter matrix 0 --|-- parameter matrix 1 --|-- parameter matrix 2 --| ...
    //! ~~~
    //!
    //! The following sections describe \ref setRNNWeightsOrder "the order of weight matrices" and
    //! \ref setRNNWeightsLayout "the layout of elements within a weight matrix".
    //!
    //! \section setRNNWeightsOrder Order of weight matrices
    //!
    //! The parameter matrices are ordered as described below:
    //!
    //! ~~~
    //!    Let G(op, l) be defined to be a function that produces lists of parameter names, as follows:
    //!
    //!         G(::kRELU, l) := [ Wl[i], Rl[i] ]
    //!         G(::kTANH, l) := [ Wl[i], Rl[i] ]
    //!         G(::kLSTM, l) := [ Wl[f], Wl[i], Wl[c], Wl[o], Rl[f], Rl[i], Rl[c], Rl[o] ]
    //!         G(::kGRU, l)  := [ Wl[z], Wl[r], Wl[h], Rl[z], Rl[r], Rl[h] ]
    //!
    //!    where Wl[g] and Rl[g] are the names of the input and recurrent
    //!    input weight matrices for gate g, layer index l.
    //!
    //!    See RNNOperation for an overview of the naming convention used for gates.
    //!
    //!    If getDirection() == ::kUNIDIRECTION, then l identifies the stacked layer of the
    //!    RNN, with l=0 being the first recurrent layer and l=L-1 being the last recurrent layer.
    //!
    //!    If getDirection() == ::kBIDIRECTION, then (l % 2) identifies the direction of the
    //!    recurrent layer (forward if 0, or backward if 1), and (l / 2) identifies the position
    //!    of the recurrent layer within the (forward or backward) stack.
    //!
    //!    Let op := getOperation(),
    //!        L  := { ::kUNIDIRECTION => getLayerCount()
    //!              { ::kBIDIRECTION => (2 * getLayerCount())
    //!
    //!    Then the ordering of parameter matrices is the list produced by concatenating
    //!    G(op, 0), G(op, 1), G(op, 2), ..., G(op, L-1).
    //! ~~~
    //!
    //! For example:
    //!
    //!    - an RNN with `getLayerCount() == 3`, `getDirection() == ::kUNIDIRECTION`,
    //!      and `getOperation() == ::kRELU` has the following order:
    //!
    //!      `[ W0[i], R0[i], W1[i], R1[i], W2[i], R2[i] ]`
    //!
    //!    - an RNN with `getLayerCount() == 2`, `getDirection() == ::kUNIDIRECTION`,
    //!      and `getOperation() == ::kGRU` has the following order:
    //!
    //!      `[ W0[z], W0[r], W0[h], R0[z], R0[r], R0[h], W1[z], W1[r], W1[h], R1[z], R1[r], R1[h] ]`
    //!
    //!    - an RNN with `getLayerCount() == 2`, `getDirection() == ::kBIDIRECTION`,
    //!      and `getOperation() == ::kRELU` has the following order:
    //!
    //!      `[ W0_fw[i], R0_fw[i], W0_bw[i], R0_bw[i], W1_fw[i], R1_fw[i], W1_bw[i], R1_bw[i] ]`
    //!
    //!      (fw = "forward", bw = "backward")
    //!
    //! \section setRNNWeightsLayout Layout of elements within a weight matrix
    //!
    //! Each parameter matrix is row-major in memory, and has the following dimensions:
    //!
    //! ~~~
    //!     Let K := { ::kUNIDIRECTION => 1
    //!              { ::kBIDIRECTION => 2
    //!         l := layer index (as described above)
    //!         H := getHiddenSize()
    //!         E := getDataLength() (the embedding length)
    //!         isW := true if the matrix is an input (W) matrix, and false if
    //!                the matrix is a recurrent input (R) matrix.
    //!
    //!    if isW:
    //!       if l < K and ::kSKIP:
    //!          (numRows, numCols) := (0, 0) # input matrix is skipped
    //!       elif l < K and ::kLINEAR:
    //!          (numRows, numCols) := (H, E) # input matrix acts on input data size E
    //!       elif l >= K:
    //!          (numRows, numCols) := (H, K * H) # input matrix acts on previous hidden state
    //!    else: # not isW
    //!       (numRows, numCols) := (H, H)
    //! ~~~
    //!
    //! In other words, the input weights of the first layer of the RNN (if
    //! not skipped) transform a `getDataLength()`-size column
    //! vector into a `getHiddenSize()`-size column vector.  The input
    //! weights of subsequent layers transform a `K*getHiddenSize()`-size
    //! column vector into a `getHiddenSize()`-size column vector.  `K=2` in
    //! the bidirectional case to account for the full hidden state being
    //! the concatenation of the forward and backward RNN hidden states.
    //!
    //! The recurrent weight matrices for all layers all have shape `(H, H)`,
    //! both in the unidirectional and bidirectional cases.  (In the
    //! bidirectional case, each recurrent weight matrix for the (forward or
    //! backward) RNN cell operates on the previous (forward or
    //! backward) RNN cell's hidden state, which is size `H`).
    //!
    //! \see getWeights(), #RNNOperation
    //!
    virtual void setWeights(Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the W weights for the RNN.
    //!
    //! \see setWeights()
    //!
    virtual Weights getWeights() const TRTNOEXCEPT = 0;

    //!
    //! \param bias The weight structure holding the bias parameters.
    //!
    //! \brief Set the bias parameters for the RNN.
    //!
    //! The trained weights for the bias parameter vectors of the RNN.
    //! The #DataType for this structure must be ::kFLOAT or ::kHALF, and must be the same
    //! datatype as the input tensor.
    //!
    //! The layout of the weight structure depends on the #RNNOperation, #RNNInputMode, and
    //! #RNNDirection of the layer.  The array specified by `weights.values` contains a sequence of
    //! bias vectors, where each bias vector is linearly appended after the previous
    //! without padding; e.g., if bias vector 0 and 1 have M and N elements respectively, then
    //! the layout of `weights.values` in memory looks like:
    //!
    //! ~~~
    //! index | 0 1 2 3 4 ...  M-2 M-1 | M M+1  ... M+N-2 M+N-1 | M+N M+N+1 M+N+2 ...   | ...
    //! data  |--   bias vector 0    --|--   bias vector 1    --|--   bias vector 2   --| ...
    //! ~~~
    //!
    //! The ordering of bias vectors is similar to the \ref setRNNWeightsOrder "ordering of weight matrices"
    //! as described in setWeights().  To determine the order of bias vectors for a given RNN configuration,
    //! determine the ordered list of weight matrices `[ W0, W1, ..., Wn ]`.  Then replace each weight matrix
    //! with its corresponding bias vector, i.e. apply the following transform (for layer `l`, gate `g`):
    //!
    //! - `Wl[g]` becomes `Wbl[g]`
    //! - `Rl[g]` becomes `Rbl[g]`
    //!
    //! For example:
    //!
    //!    - an RNN with `getLayerCount() == 3`, `getDirection() == ::kUNIDIRECTION`,
    //!      and `getOperation() == ::kRELU` has the following order:
    //!
    //!      `[ Wb0[i], Rb0[i], Wb1[i], Rb1[i], Wb2[i], Rb2[i] ]`
    //!
    //!    - an RNN with `getLayerCount() == 2`, `getDirection() == ::kUNIDIRECTION`,
    //!      and `getOperation() == ::kGRU` has the following order:
    //!
    //!      `[ Wb0[z], Wb0[r], Wb0[h], Rb0[z], Rb0[r], Rb0[h], Wb1[z], Wb1[r], Wb1[h], Rb1[z], Rb1[r], Rb1[h] ]`
    //!
    //!    - an RNN with `getLayerCount() == 2`, `getDirection() == ::kBIDIRECTION`,
    //!      and `getOperation() == ::kRELU` has the following order:
    //!
    //!      `[ Wb0_fw[i], Rb0_fw[i], Wb0_bw[i], Rb0_bw[i], Wb1_fw[i], Rb1_fw[i], Wb1_bw[i], Rb1_bw[i] ]`
    //!
    //!      (fw = "forward", bw = "backward")
    //!
    //! Each bias vector has a fixed size, getHiddenSize().
    //!
    //! \see getBias(), #RNNOperation
    //!
    virtual void setBias(Weights bias) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the bias parameter vector for the RNN.
    //!
    //! \see setBias()
    //!
    virtual Weights getBias() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the length of the data being processed by the RNN for use in computing
    //! other values.
    //!
    //! \see setHiddenState(), setCellState()
    //!
    virtual int getDataLength() const TRTNOEXCEPT = 0;

    //!
    //! \param hidden The initial hidden state of the RNN.
    //!
    //! \brief Set the initial hidden state of the RNN with the provided \p hidden ITensor.
    //!
    //! The layout for \p hidden is a linear layout of a 3D matrix:
    //!  - C - The number of layers in the RNN, it must match getLayerCount().
    //!  - H - The number of mini-batches for each time sequence.
    //!  - W - The size of the per layer hidden states, it must match getHiddenSize().
    //!
    //! If getDirection() is ::kBIDIRECTION, the amount of space required is doubled and C is equal to
    //! getLayerCount() * 2.
    //!
    //! If hidden is not specified, then the initial hidden state is set to zero.
    //!
    //! \see getHiddenState()
    //!
    virtual void setHiddenState(ITensor& hidden) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the initial hidden state of the RNN.
    //!
    //! \return nullptr if no initial hidden tensor was specified, the initial hidden data otherwise.
    //!
    virtual ITensor* getHiddenState() const TRTNOEXCEPT = 0;

    //!
    //! \param cell The initial cell state of the RNN.
    //!
    //! \brief Set the initial cell state of the RNN with the provided \p cell ITensor.
    //!
    //! The layout for \p cell is a linear layout of a 3D matrix:
    //!  - C - The number of layers in the RNN, it must match getLayerCount().
    //!  - H - The number of mini-batches for each time sequence.
    //!  - W - The size of the per layer hidden states, it must match getHiddenSize().
    //!
    //! If \p cell is not specified, then the initial cell state is set to zero.
    //!
    //! If getDirection() is ::kBIDIRECTION, the amount of space required is doubled and C is equal to
    //! getLayerCount() * 2.
    //!
    //! The cell state only affects LSTM RNN's.
    //!
    //! \see getCellState()
    //!
    virtual void setCellState(ITensor& cell) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the initial cell state of the RNN.
    //!
    //! \return nullptr if no initial cell tensor was specified, the initial cell data otherwise.
    //!
    virtual ITensor* getCellState() const TRTNOEXCEPT = 0;

protected:
    virtual ~IRNNLayer() {}
};

//!
//! \enum RNNGateType
//!
//! \brief Identifies an individual gate within an RNN cell.
//!
//! \see RNNOperation
//!
enum class RNNGateType : int
{
    kINPUT = 0,  //!< Input gate  (i).
    kOUTPUT = 1, //!< Output gate (o).
    kFORGET = 2, //!< Forget gate (f).
    kUPDATE = 3, //!< Update gate (z).
    kRESET = 4,  //!< Reset gate  (r).
    kCELL = 5,   //!< Cell gate   (c).
    kHIDDEN = 6  //!< Hidden gate (h).
};

template <>
constexpr inline int EnumMax<RNNGateType>()
{
    return 7;
} //!< Maximum number of elements in RNNGateType enum. \see RNNGateType

//!
//! \class IRNNv2Layer
//!
//! \brief An RNN layer in a network definition, version 2.
//!
//! This layer supersedes IRNNLayer.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRNNv2Layer : public ILayer
{
public:
    virtual int32_t getLayerCount() const TRTNOEXCEPT = 0;   //< Get the layer count of the RNN
    virtual int32_t getHiddenSize() const TRTNOEXCEPT = 0;   //< Get the hidden size of the RNN
    virtual int32_t getMaxSeqLength() const TRTNOEXCEPT = 0; //< Get the maximum sequence length of the RNN
    virtual int32_t getDataLength() const TRTNOEXCEPT = 0;   //< Get the maximum data length of the RNN

    //!
    //! \brief Specify individual sequence lengths in the batch with the ITensor pointed to by
    //! \p seqLengths.
    //!
    //! The \p seqLengths ITensor should be a {N1, ..., Np} tensor, where N1..Np are the index dimensions
    //! of the input tensor to the RNN.
    //!
    //! If this is not specified, then the RNN layer assumes all sequences are size getMaxSeqLength().
    //!
    //! All sequence lengths in \p seqLengths should be in the range [1, getMaxSeqLength()].  Zero-length
    //! sequences are not supported.
    //!
    //! This tensor must be of type DataType::kINT32.
    //!
    virtual void setSequenceLengths(ITensor& seqLengths) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the sequence lengths specified for the RNN.
    //!
    //! \return nullptr if no sequence lengths were specified, the sequence length data otherwise.
    //!
    //! \see setSequenceLengths()
    //!
    virtual ITensor* getSequenceLengths() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the operation of the RNN layer.
    //! \see getOperation(), RNNOperation
    //!
    virtual void setOperation(RNNOperation op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the operation of the RNN layer.
    //! \see setOperation(), RNNOperation
    //!
    virtual RNNOperation getOperation() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the input mode of the RNN layer.
    //! \see getInputMode(), RNNInputMode
    //!
    virtual void setInputMode(RNNInputMode op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the input mode of the RNN layer.
    //! \see setInputMode(), RNNInputMode
    //!
    virtual RNNInputMode getInputMode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the direction of the RNN layer.
    //! \see getDirection(), RNNDirection
    //!
    virtual void setDirection(RNNDirection op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the direction of the RNN layer.
    //! \see setDirection(), RNNDirection
    //!
    virtual RNNDirection getDirection() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the weight parameters for an individual gate in the RNN.
    //!
    //! \param layerIndex The index of the layer that contains this gate.  See the section
    //!        \ref setRNNWeightsOrder "Order of weight matrices" in IRNNLayer::setWeights()
    //!        for a description of the layer index.
    //! \param gate The name of the gate within the RNN layer.  The gate name must correspond
    //!        to one of the gates used by this layer's #RNNOperation.
    //! \param isW True if the weight parameters are for the input matrix W[g]
    //!        and false if they are for the recurrent input matrix R[g].  See
    //!        #RNNOperation for equations showing how these matrices are used
    //!        in the RNN gate.
    //! \param weights The weight structure holding the weight parameters, which are stored
    //!        as a row-major 2D matrix.  See \ref setRNNWeightsLayout "the layout of elements within a weight matrix"
    //!        in IRNNLayer::setWeights() for documentation on the expected
    //!        dimensions of this matrix.
    //!
    virtual void setWeightsForGate(int layerIndex, RNNGateType gate, bool isW, Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the weight parameters for an individual gate in the RNN.
    //! \see setWeightsForGate()
    //!
    virtual Weights getWeightsForGate(int layerIndex, RNNGateType gate, bool isW) const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the bias parameters for an individual gate in the RNN.
    //!
    //! \param layerIndex The index of the layer that contains this gate.  See the section
    //!        \ref setRNNWeightsOrder "Order of weight matrices" in IRNNLayer::setWeights()
    //!        for a description of the layer index.
    //! \param gate The name of the gate within the RNN layer.  The gate name must correspond
    //!        to one of the gates used by this layer's #RNNOperation.
    //! \param isW True if the bias parameters are for the input bias Wb[g]
    //!        and false if they are for the recurrent input bias Rb[g].  See
    //!        #RNNOperation for equations showing how these bias vectors are used
    //!        in the RNN gate.
    //! \param bias The weight structure holding the bias parameters, which should be an
    //!        array of size getHiddenSize().
    //!
    virtual void setBiasForGate(int layerIndex, RNNGateType gate, bool isW, Weights bias) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the bias parameters for an individual gate in the RNN.
    //! \see setBiasForGate()
    //!
    virtual Weights getBiasForGate(int layerIndex, RNNGateType gate, bool isW) const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the initial hidden state of the RNN with the provided \p hidden ITensor.
    //!
    //! The \p hidden ITensor should have the dimensions `{N1, ..., Np, L, H}`, where:
    //!
    //!  - `N1..Np` are the index dimensions specified by the input tensor
    //!  - `L` is the number of layers in the RNN, equal to getLayerCount() if getDirection is ::kUNIDIRECTION,
    //!     and 2x getLayerCount() if getDirection is ::kBIDIRECTION. In the bi-directional
    //!     case, layer `l`'s final forward hidden state is stored in `L = 2*l`, and
    //!     final backward hidden state is stored in `L= 2*l + 1`.
    //!  - `H` is the hidden state for each layer, equal to getHiddenSize().
    //!
    virtual void setHiddenState(ITensor& hidden) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the initial hidden state of the RNN.
    //! \see setHiddenState()
    //!
    virtual ITensor* getHiddenState() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the initial cell state of the LSTM with the provided \p cell ITensor.
    //!
    //! The \p cell ITensor should have the dimensions `{N1, ..., Np, L, H}`, where:
    //!
    //!  - `N1..Np` are the index dimensions specified by the input tensor
    //!  - `L` is the number of layers in the RNN, equal to getLayerCount() if getDirection is ::kUNIDIRECTION,
    //!     and 2x getLayerCount() if getDirection is ::kBIDIRECTION. In the bi-directional
    //!     case, layer `l`'s final forward hidden state is stored in `L = 2*l`, and
    //!     final backward hidden state is stored in `L= 2*l + 1`.
    //!  - `H` is the hidden state for each layer, equal to getHiddenSize().
    //!
    //! It is an error to call setCellState() on an RNN layer that is not configured with RNNOperation::kLSTM.
    //!
    virtual void setCellState(ITensor& cell) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the initial cell state of the RNN.
    //! \see setCellState()
    //!
    virtual ITensor* getCellState() const TRTNOEXCEPT = 0;

protected:
    virtual ~IRNNv2Layer() {}
};

//!
//! \class IOutputDimensionsFormula
//!
//! \brief Application-implemented interface to compute layer output sizes.
//!
class IOutputDimensionsFormula
{
public:
    //!
    //! \brief Application-implemented interface to compute the HW output dimensions of a layer from the layer input
    //! and parameters.
    //!
    //! \param inputDims The input dimensions of the layer.
    //! \param kernelSize The kernel size (or window size, for a pooling layer) parameter of the layer operation.
    //! \param stride The stride parameter for the layer.
    //! \param padding The padding parameter of the layer.
    //! \param dilation The dilation parameter of the layer (only applicable to convolutions).
    //! \param layerName The name of the layer.
    //!
    //! \return The output size of the layer
    //!
    //! Note that for dilated convolutions, the dilation is applied to the kernel size before this routine is called.
    //!
    virtual DimsHW compute(DimsHW inputDims, DimsHW kernelSize, DimsHW stride, DimsHW padding, DimsHW dilation, const char* layerName) const TRTNOEXCEPT = 0;

    virtual ~IOutputDimensionsFormula() {}
};

//!
//! \class IPluginLayer
//!
//! \brief Layer type for plugins.
//!
//! \see IPluginExt
//!
//! \deprecated This interface is superseded by IPluginV2Layer
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class TRT_DEPRECATED IPluginLayer : public ILayer
{
public:
    //!
    //! \brief Get the plugin for the layer.
    //!
    //! \see IPluginExt
    //!
    virtual IPlugin& getPlugin() TRTNOEXCEPT = 0;

protected:
    virtual ~IPluginLayer() {}
};

//!
//! \class IPluginV2Layer
//!
//! \brief Layer type for pluginV2
//!
//! \see IPluginV2
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPluginV2Layer : public ILayer
{
public:
    //!
    //! \brief Get the plugin for the layer.
    //!
    //! \see IPluginV2
    //!
    virtual IPluginV2& getPlugin() TRTNOEXCEPT = 0;

protected:
    virtual ~IPluginV2Layer() {}
};

//!
//! \enum UnaryOperation
//!
//! \brief Enumerates the unary operations that may be performed by a Unary layer.
//!
//! \see IUnaryLayer
//!
enum class UnaryOperation : int
{
    kEXP = 0,    //!< Exponentiation.
    kLOG = 1,    //!< Log (base e).
    kSQRT = 2,   //!< Square root.
    kRECIP = 3,  //!< Reciprocal.
    kABS = 4,    //!< Absolute value.
    kNEG = 5,    //!< Negation.
    kSIN = 6,    //!< Sine.
    kCOS = 7,    //!< Cosine.
    kTAN = 8,    //!< Tangent.
    kSINH = 9,   //!< Hyperbolic sine.
    kCOSH = 10,  //!< Hyperbolic cosine.
    kASIN = 11,  //!< Inverse sine.
    kACOS = 12,  //!< Inverse cosine.
    kATAN = 13,  //!< Inverse tangent.
    kASINH = 14, //!< Inverse hyperbolic sine.
    kACOSH = 15, //!< Inverse hyperbolic cosine.
    kATANH = 16, //!< Inverse hyperbolic tangent.
    kCEIL = 17,  //!< Ceiling.
    kFLOOR = 18  //!< Floor.
};

template <>
constexpr inline int EnumMax<UnaryOperation>()
{
    return 19;
} //!< Maximum number of elements in UnaryOperation enum. \see UnaryOperation

//!
//! \class IUnaryLayer
//!
//! \brief Layer that represents an unary operation.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IUnaryLayer : public ILayer
{
public:
    //!
    //! \brief Set the unary operation for the layer.
    //!
    //! \see getOperation(), UnaryOperation
    //!
    virtual void setOperation(UnaryOperation op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the unary operation for the layer.
    //!
    //! \see setOperation(), UnaryOperation
    //!
    virtual UnaryOperation getOperation() const TRTNOEXCEPT = 0;

protected:
    virtual ~IUnaryLayer() {}
};

//!
//! \enum ReduceOperation
//!
//! \brief Enumerates the reduce operations that may be performed by a Reduce layer.
//!
enum class ReduceOperation : int
{
    kSUM = 0,
    kPROD = 1,
    kMAX = 2,
    kMIN = 3,
    kAVG = 4
};

template <>
constexpr inline int EnumMax<ReduceOperation>()
{
    return 5;
} //!< Maximum number of elements in ReduceOperation enum. \see ReduceOperation

//!
//! \class IReduceLayer
//!
//! \brief Layer that represents a reduction operator.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IReduceLayer : public ILayer
{
public:
    //!
    //! \brief Set the reduce operation for the layer.
    //!
    //! \see getOperation(), ReduceOperation
    //!
    virtual void setOperation(ReduceOperation op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the reduce operation for the layer.
    //!
    //! \see setOperation(), ReduceOperation
    //!
    virtual ReduceOperation getOperation() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the axes over which to reduce.
    //!
    //! \see getReduceAxes
    //!
    virtual void setReduceAxes(uint32_t reduceAxes) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the axes over which to reduce for the layer.
    //!
    //! \see setReduceAxes
    //!
    virtual uint32_t getReduceAxes() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the boolean that specifies whether or not to keep the reduced dimensions for the layer.
    //!
    //! \see getKeepDimensions
    //!
    virtual void setKeepDimensions(bool keepDimensions) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the boolean that specifies whether or not to keep the reduced dimensions for the layer.
    //!
    //! \see setKeepDimensions
    //!
    virtual bool getKeepDimensions() const TRTNOEXCEPT = 0;

protected:
    virtual ~IReduceLayer() {}
};

//!
//! \class IPaddingLayer
//!
//! \brief Layer that represents a padding operation.
//!
//! The padding layer adds zero-padding at the start and end of the input tensor. It only supports padding along the two
//! innermost dimensions. Applying negative padding results in cropping of the input.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IPaddingLayer : public ILayer
{
public:
    //!
    //! \brief Set the padding that is applied at the start of the tensor.
    //!
    //! Negative padding results in trimming the edge by the specified amount
    //!
    //! \see getPrePadding
    //!
    virtual void setPrePadding(DimsHW padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the padding that is applied at the start of the tensor.
    //!
    //! \see setPrePadding
    //!
    virtual DimsHW getPrePadding() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the padding that is applied at the end of the tensor.
    //!
    //! Negative padding results in trimming the edge by the specified amount
    //!
    //! \see getPostPadding
    //!
    virtual void setPostPadding(DimsHW padding) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the padding that is applied at the end of the tensor.
    //!
    //! \see setPostPadding
    //!
    virtual DimsHW getPostPadding() const TRTNOEXCEPT = 0;

protected:
    virtual ~IPaddingLayer() {}
};

struct Permutation
{
    //!
    //! The elements of the permutation.
    //! The permutation is applied as outputDimensionIndex = permutation.order[inputDimensionIndex], so to
    //! permute from CHW order to HWC order, the required permutation is [1, 2, 0], and to permute
    //! from HWC to CHW, the required permutation is [2, 0, 1].
    //!
    int order[Dims::MAX_DIMS];
};

//! \class IShuffleLayer
//!
//! \brief Layer type for shuffling data.
//!
//! This class shuffles data by applying in sequence: a transpose operation, a reshape operation
//! and a second transpose operation. The dimension types of the output are those of the reshape dimension.
//!
//! The layer has an optional second input.  If present, it must be a 1D Int32 shape tensor,
//! and the reshape dimensions are taken from it.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IShuffleLayer : public ILayer
{
public:
    //!
    //! \brief Set the permutation applied by the first transpose operation.
    //!
    //! \param permutation The dimension permutation applied before the reshape.
    //!
    //! The default is the identity permutation.
    //!
    //! \see getFirstTranspose
    //!
    virtual void setFirstTranspose(Permutation permutation) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the permutation applied by the first transpose operation.
    //!
    //! \return The dimension permutation applied before the reshape.
    //!
    //! \see setFirstTranspose
    //!
    virtual Permutation getFirstTranspose() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the reshaped dimensions.
    //!
    //! \param dimensions The reshaped dimensions.
    //!
    //! Two special values can be used as dimensions.
    //!
    //! Value 0 copies the corresponding dimension from input. This special value
    //! can be used more than once in the dimensions. If number of reshape
    //! dimensions is less than input, 0s are resolved by aligning the most
    //! significant dimensions of input.
    //!
    //! Value -1 infers that particular dimension by looking at input and rest
    //! of the reshape dimensions. Note that only a maximum of one dimension is
    //! permitted to be specified as -1.
    //!
    //! The product of the new dimensions must be equal to the product of the old.
    //!
    //! If there is a second input, i.e. reshape dimensions are dynamic,
    //! calling setReshapeDimensions() is an error and does not update
    //! the dimensions.
    //!
    virtual void setReshapeDimensions(Dims dimensions) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the reshaped dimensions.
    //!
    //! \return The reshaped dimensions.
    //!
    //! If there is a second input, returns Dims with nbDims == -1.
    //!
    virtual Dims getReshapeDimensions() const TRTNOEXCEPT = 0;

    //!
    //! \brief Relaxes ILayer::setInput to allow appending a second input.
    //!
    //! Like ILayer::setInput, but additionally works if index==1, nbInputs()==1, and
    //! there is no implicit batch dimension, in which case nbInputs() changes to 2.
    //!
    //! When there is a 2nd input, the reshapeDimensions are taken from it, overriding
    //! the dimensions supplied by setReshapeDimensions.
    //!
    void setInput(int index, ITensor& tensor) _TENSORRT_OVERRIDE TRTNOEXCEPT = 0;

    //!
    //! \brief Set the permutation applied by the second transpose operation.
    //!
    //! \param permutation The dimension permutation applied after the reshape.
    //!
    //! The default is the identity permutation.
    //!
    //! The permutation is applied as outputDimensionIndex = permutation.order[inputDimensionIndex], so to
    //! permute from CHW order to HWC order, the required permutation is [1, 2, 0].
    //!
    //! \see getSecondTranspose
    //!
    virtual void setSecondTranspose(Permutation permutation) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the permutation applied by the second transpose operation.
    //!
    //! \return The dimension permutation applied after the reshape.
    //!
    //! \see setSecondTranspose
    //!
    virtual Permutation getSecondTranspose() const TRTNOEXCEPT = 0;

protected:
    virtual ~IShuffleLayer() {}
};

//!
//! \brief Slices an input tensor into an output tensor based on the offset and strides.
//!
//! The slice layer has two variants, static and dynamic. Static slice specifies the start, size, and stride
//! dimensions at layer create time via Dims and can use the get/set accessor functions of the ISliceLayer. Dynamic
//! slice specifies the start and size dimensions at layer create time via ITensors and uses ILayer::setTensor to
//! set the optional stride parameter after layer construction.
//! An application can determine if the ISliceLayer is dynamic or static based on if there are 3 or 4 inputs(Dynamic)
//! or 1 input(Static). When working on a shape tensor, a dynamic slace layer must have start, size, and stride
//! specified at build time.
//!
//! The slice layer selects for each dimension a start location from within the input tensor, and given the
//! specified stride, copies strided elements to the output tensor. Start, Size, and Stride shape tensors must be
//! DataType::kINT32.
//!
//! For example using slice on a data tensor:
//! input = {{0, 1}, {2, 3}, {4, 5}}
//! start = {1, 0}
//! size = {1, 2}
//! stride = {1, 2}
//! output = {1, 5}
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ISliceLayer : public ILayer
{
public:
    //!
    //! \brief Set the start offset that the slice layer uses to create the output slice.
    //!
    //! \param start The start offset to read data from the input tensor.
    //!
    //! If the SliceLayer is using dynamic inputs for the start parameter, calling setStart() results in an error
    //! and does not update the dimensions.
    //!
    //! \see getStart
    //!
    virtual void setStart(Dims start) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the start offset for the slice layer.
    //!
    //! \return The start offset, or an invalid Dims structure.
    //!
    //! If the SliceLayer is using dynamic inputs for the start parameter, this function returns an invalid
    //! Dims structure.
    //!
    //! \see setStart
    //!
    virtual Dims getStart() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the dimensions of the output slice.
    //!
    //! \param size The dimensions of the output slice.
    //!
    //! If the SliceLayer is using dynamic inputs for the size parameter, calling setSize() results in an error
    //! and does not update the dimensions.
    //!
    //! \see getSize
    //!
    virtual void setSize(Dims size) TRTNOEXCEPT = 0;

    //!
    //! \brief Get dimensions of the output slice.
    //!
    //! \return The output dimension, or an invalid Dims structure.
    //!
    //! If the SliceLayer is using dynamic inputs for the size parameter, this function returns an invalid
    //! Dims structure.
    //!
    //! \see setSize
    //!
    virtual Dims getSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the stride for computing the output slice data.
    //!
    //! \param stride The dimensions of the stride to compute the values to store in the output slice.
    //!
    //! If the SliceLayer is using dynamic inputs for the stride parameter, calling setSlice() results in an error
    //! and does not update the dimensions.
    //!
    //! \see getStride
    //!
    virtual void setStride(Dims stride) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the stride for the output slice.
    //!
    //! \return The slicing stride, or an invalid Dims structure.
    //!
    //! If the SliceLayer is using dynamic inputs for the stride parameter, this function returns a invalid
    //! Dims structure.
    //!
    //! \see setStride
    //!
    virtual Dims getStride() const TRTNOEXCEPT = 0;

    //!
    //! \brief replace an input of this layer with a specific tensor.
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor
    //!
    //! Sets the input tensor for the given index. The index must be 0 for a static slice layer.
    //! A static slice layer is converted to a dynamic slice layer by calling setInput with an index > 0.
    //! A dynamic slice layer cannot be converted back to a static slice layer.
    //!
    //! For a dynamic slice layer, the values 0-3 are valid. If an index > 0 is specified, all values between
    //! index 0 and that index must be dynamic tensors. The values larger than index can use static dimensions.
    //! For example, if an index of two is specified, the stride tensor can be set via setStride, but the start tensor
    //! must be specified via setInput as both size and start are converted to dynamic tensors.
    //! The indices in the dynamic case are as follows:
    //!
    //! Index | Description
    //!   0   | Data or Shape tensor to be sliced.
    //!   1   | The start tensor to begin slicing, N-dimensional for Data, and 1-D for Shape.
    //!   2   | The size tensor of the resulting slice, N-dimensional for Data, and 1-D for Shape.
    //!   3   | The stride of the slicing operation, N-dimensional for Data, and 1-D for Shape.
    //!
    //! If this function is called with a value greater than 0, then the function getNbInputs() changes
    //! from returning 1 to index + 1. When converting from static to dynamic slice layer,
    //! all unset tensors, between 1 and index + 1, are initialized to nullptr. It is an error to attempt to build
    //! a network that has any nullptr inputs.
    //!
    void setInput(int index, ITensor& tensor) _TENSORRT_OVERRIDE TRTNOEXCEPT = 0;

protected:
    virtual ~ISliceLayer() {}
};

//! \class IShapeLayer
//!
//! \brief Layer type for getting shape of a tensor.
//!
//! This class sets the output to a one-dimensional tensor with the dimensions of the input tensor.
//!
//! For example, if the input is a four-dimensional tensor (of any type) with
//! dimensions [2,3,5,7], the output tensor is a one-dimensional Int32 tensor
//! of length 4 containing the sequence 2, 3, 5, 7.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IShapeLayer : public ILayer
{
protected:
    virtual ~IShapeLayer() {}
};

//!
//! \enum TopKOperation
//!
//! \brief Enumerates the operations that may be performed by a TopK layer.
//!
enum class TopKOperation : int
{
    kMAX = 0, //!< Maximum of the elements.
    kMIN = 1, //!< Minimum of the elements.
};

template <>
constexpr inline int EnumMax<TopKOperation>()
{
    return 2;
} //!< Maximum number of elements in TopKOperation enum. \see TopKOperation

//!
//! \class ITopKLayer
//!
//! \brief Layer that represents a TopK reduction.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class ITopKLayer : public ILayer
{
public:
    //!
    //! \brief Set the operation for the layer.
    //!
    //! \see getOperation(), TopKOperation
    //!
    virtual void setOperation(TopKOperation op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the operation for the layer.
    //!
    //! \see setOperation(), TopKOperation
    //!
    virtual TopKOperation getOperation() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the k value for the layer.
    //!
    //! Currently only values up to 25 are supported.
    //!
    //! \see getK()
    //!
    virtual void setK(int k) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the k value for the layer.
    //!
    //! \see setK()
    //!
    virtual int getK() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set which axes to reduce for the layer.
    //!
    //! \see getReduceAxes()
    //!
    virtual void setReduceAxes(uint32_t reduceAxes) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the axes to reduce for the layer.
    //!
    //! \see setReduceAxes()
    //!
    virtual uint32_t getReduceAxes() const TRTNOEXCEPT = 0;

protected:
    virtual ~ITopKLayer() {}
};

//!
//! \enum MatrixOperation
//!
//! \brief Enumerates the operations that may be performed on a tensor
//!        by IMatrixMultiplyLayer before multiplication.
//!
enum class MatrixOperation : int
{
    //! Treat x as a matrix if it has two dimensions, or as a collection of
    //! matrices if x has more than two dimensions, where the last two dimensions
    //! are the matrix dimensions.  x must have at least two dimensions.
    kNONE,

    //! Like kNONE, but transpose the matrix dimensions.
    kTRANSPOSE,

    //! Treat x as a vector if it has one dimension, or as a collection of
    //! vectors if x has more than one dimension.  x must have at least one dimension.
    kVECTOR
};

template <>
constexpr inline int EnumMax<MatrixOperation>()
{
    return 3;
} //!< Maximum number of elements in MatrixOperation enum. \see DataType

//!
//! \class IMatrixMultiplyLayer
//!
//! \brief Layer that represents a Matrix Multiplication.
//!
//! Let A be op(getInput(0)) and B be op(getInput(1)) where
//! op(x) denotes the corresponding MatrixOperation.
//!
//! When A and B are matrices or vectors, computes the inner product A * B:
//!
//!     matrix * matrix -> matrix
//!     matrix * vector -> vector
//!     vector * matrix -> vector
//!     vector * vector -> scalar
//!
//! Inputs of higher rank are treated as collections of matrices or vectors.
//! The output will be a corresponding collection of matrices, vectors, or scalars.
//!
//! For a dimension that is not one of the matrix or vector dimensions:
//! If the dimension is 1 for one of the tensors but not the other tensor,
//! the former tensor is broadcast along that dimension to match the dimension of the latter tensor.
//! The number of these extra dimensions for A and B must match.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IMatrixMultiplyLayer : public ILayer
{
public:
    //!
    //! \brief Set the operation for an input tensor.
    //! \param index Input tensor number (0 or 1).
    //! \param op New operation.
    //! \see getTranspose()
    //!
    virtual void setOperation(int index, MatrixOperation op) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the operation for an input tensor.
    //! \param index Input tensor number (0 or 1).
    //! \see setTranspose()
    //!
    virtual MatrixOperation getOperation(int index) const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the transpose flag for an input tensor.
    //! \param index Input tensor number (0 or 1).
    //! \param val New transpose flag.
    //! \see getTranspose()
    //!
    //! \deprecated setTranspose is superseded by setOperation.
    //!
    TRT_DEPRECATED virtual void setTranspose(int index, bool val) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the transpose flag for an input tensor.
    //! \param index Input tensor number (0 or 1).
    //! \see setTranspose()
    //!
    //! \deprecated getTranspose is superseded by getOperation.
    //!
    TRT_DEPRECATED virtual bool getTranspose(int index) const TRTNOEXCEPT = 0;

protected:
    virtual ~IMatrixMultiplyLayer() {}
};

//!
//! \class IRaggedSoftMaxLayer
//!
//! \brief A RaggedSoftmax layer in a network definition.
//!
//! This layer takes a ZxS input tensor and an additional Zx1 bounds tensor
//! holding the lengths of the Z sequences.
//!
//! This layer computes a softmax across each of the Z sequences.
//!
//! The output tensor is of the same size as the input tensor.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IRaggedSoftMaxLayer : public ILayer
{
protected:
    virtual ~IRaggedSoftMaxLayer() {}
};

//! \class IIdentityLayer
//!
//! \brief A layer that represents the identity function.
//!
//! If tensor precision is being explicitly specified, it can be used to transform from one precision to another.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IIdentityLayer : public ILayer
{
protected:
    virtual ~IIdentityLayer() {}
};

//! \class IConstantLayer
//!
//! \brief Layer that represents a constant value.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IConstantLayer : public ILayer
{
public:
    //!
    //! \brief Set the weights for the layer.
    //!
    //! If weights.type is DataType::kINT32, the output is a tensor of 32-bit indices.
    //! Otherwise the output is a tensor of real values and the output type will be
    //! follow TensorRT's normal precision rules.
    //!
    //! \see getWeights()
    //!
    virtual void setWeights(Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the weights for the layer.
    //!
    //! \see setWeights
    //!
    virtual Weights getWeights() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the dimensions for the layer.
    //!
    //! \param dimensions The dimensions of the layer
    //!
    //! \see setDimensions
    //!
    virtual void setDimensions(Dims dimensions) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the dimensions for the layer.
    //!
    //! \return the dimensions for the layer
    //!
    //! \see getDimensions
    //!
    virtual Dims getDimensions() const TRTNOEXCEPT = 0;

protected:
    virtual ~IConstantLayer() {}
};

//!
//! \class IParametricReLULayer
//!
//! \brief Layer that represents a parametric ReLU operation.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IParametricReLULayer : public ILayer
{
protected:
    virtual ~IParametricReLULayer() noexcept {}
};

//! \enum ResizeMode
//!
//! \brief Enumerates various modes of resize in the resize layer.
//!        Resize mode set using setResizeMode().
//!
enum class ResizeMode : int
{
    kNEAREST = 0, // N-D (0 < N <= 8) nearest neighbor resizing.
    kLINEAR = 1   // Can handle linear (1D), bilinear (2D), and trilinear (3D) resizing.
};

template <>
constexpr inline int EnumMax<ResizeMode>()
{
    return 2;
} //!< Maximum number of elements in ResizeMode enum. \see ResizeMode

//! \class IResizeLayer
//!
//! \brief A resize layer in a network definition.
//!
//! Resize layer can be used for resizing a N-D tensor.
//!
//! Resize layer currently supports the following configurations:
//!     -   ResizeMode::kNEAREST - resizes innermost `m` dimensions of N-D, where 0 < m <= min(8, N) and N > 0
//!     -   ResizeMode::kLINEAR - resizes innermost `m` dimensions of N-D, where 0 < m <= min(3, N) and N > 0
//!
//! Default resize mode is ResizeMode::kNEAREST.
//! Resize layer provides two ways to resize tensor dimensions.
//!     -   Set output dimensions directly. It can be done for static as well as dynamic resize layer.
//!         Static resize layer requires output dimensions to be known at build-time.
//!         Dynamic resize layer requires output dimensions to be set as one of the input tensors.
//!     -   Set scales for resize. Each output dimension is calculated as floor(input dimension * scale).
//!         Only static resize layer allows setting scales where the scales are known at build-time.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IResizeLayer : public ILayer
{
public:
    //!
    //! \brief Set the output dimensions.
    //!
    //! \param dimensions The output dimensions. Number of output dimensions must be the same as the number of input dimensions.
    //!
    //! If there is a second input, i.e. resize layer is dynamic,
    //! calling setOutputDimensions() is an error and does not update the
    //! dimensions.
    //!
    //! Output dimensions can be specified directly, or via scale factors relative to input dimensions.
    //! Scales for resize can be provided using setScales().
    //!
    //! \see setScales
    //! \see getOutputDimensions
    //!
    virtual void setOutputDimensions(Dims dimensions) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the output dimensions.
    //!
    //! \return The output dimensions.
    //!
    virtual Dims getOutputDimensions() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the resize scales.
    //!
    //! \param scales An array of resize scales.
    //! \param nbScales Number of scales. Number of scales must be equal to the number of input dimensions.
    //!
    //! If there is a second input, i.e. resize layer is dynamic,
    //! calling setScales() is an error and does not update the scales.
    //!
    //! Output dimensions are calculated as follows:
    //! outputDims[i] = floor(inputDims[i] * scales[i])
    //!
    //! Output dimensions can be specified directly, or via scale factors relative to input dimensions.
    //! Output dimensions can be provided directly using setOutputDimensions().
    //!
    //! \see setOutputDimensions
    //! \see getScales
    //!
    virtual void setScales(const float* scales, int nbScales) TRTNOEXCEPT = 0;

    //!
    //! \brief Copies resize scales to scales[0, ..., nbScales-1], where nbScales is the number of scales that were set.
    //!
    //! \param size The number of scales to get. If size != nbScales, no scales will be copied.
    //!
    //! \param scales Pointer to where to copy the scales. Scales will be copied only if
    //!               size == nbScales and scales != nullptr.
    //!
    //! In case the size is not known consider using size = 0 and scales = nullptr. This method will return
    //! the number of resize scales.
    //!
    //! \return The number of resize scales i.e. nbScales if scales were set.
    //!         Return -1 in case no scales were set or resize layer is used in dynamic mode.
    //!
    virtual int getScales(int size, float* scales) const TRTNOEXCEPT = 0;

    //!
    //! \brief Set resize mode for an input tensor.
    //!
    //! Supported resize modes are Nearest Neighbor and Linear.
    //!
    //! \see ResizeMode
    //!
    virtual void setResizeMode(ResizeMode resizeMode) TRTNOEXCEPT = 0;

    //!
    //! \brief Get resize mode for an input tensor.
    //!
    //! \return The resize mode.
    //!
    virtual ResizeMode getResizeMode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set whether to align corners while resizing.
    //!
    //! If true, the centers of the 4 corner pixels of both input and output
    //! tensors are aligned i.e. preserves the values of corner
    //! pixels.
    //!
    //! Default: false.
    //!
    virtual void setAlignCorners(bool alignCorners) TRTNOEXCEPT = 0;

    //!
    //! \brief True if align corners has been set.
    //!
    //! \return True if align corners has been set, false otherwise.
    //!
    virtual bool getAlignCorners() const TRTNOEXCEPT = 0;

    //!
    //! \brief Relaxes ILayer::setInput to allow appending a second input.
    //!
    //! \param index the index of the input to modify.
    //! \param tensor the new input tensor.
    //!
    //! Like ILayer::setInput, but additionally works if index == 1 and nbInputs == 1, and
    //! there is no implicit batch dimension, in which case nbInputs() changes to 2.
    //! Once such additional input is set, resize layer works in dynamic mode.
    //!
    //! When index == 1 and nbInputs == 1, the output dimensions are used from
    //! the input tensor, overriding the dimensions supplied by setOutputDimensions.
    //!
    //! \warning tensor must be a shape tensor.
    //!
    void setInput(int index, ITensor& tensor) _TENSORRT_OVERRIDE TRTNOEXCEPT = 0;

protected:
    virtual ~IResizeLayer() {}
};

//!
//! \class INetworkDefinition
//!
//! \brief A network definition for input to the builder.
//!
//! A network definition defines the structure of the network, and combined with a INetworkConfig, is built
//! into an engine using an IBuilder. An INetworkDefinition can either have an implicit batch dimensions, specified
//! at runtime, or all dimensions explicit, full dims mode, in the network definition. When a network has been
//! created using createNetwork(), only implicit batch size mode is supported. The function hasImplicitBatchSize()
//! is used to query the mode of the network.
//!
//! A network with implicit batch dimensions returns the dimensions of a layer without the implicit dimension,
//! and instead the batch is specified at execute/enqueue time. If the network has all dimensions specified, then
//! the first dimension follows elementwise broadcast rules: if it is 1 for some inputs and is some value N for all
//! other inputs, then the first dimension of each outut is N, and the inputs with 1 for the first dimension are
//! broadcast. Having divergent batch sizes across inputs to a layer is not supported.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class INetworkDefinition
{
public:
    //!
    //! \brief Add an input tensor to the network.
    //!
    //! The name of the input tensor is used to find the index into the buffer array for an engine built from
    //! the network. The volume of the dimensions must be less than 2^30 elements.

    //! For networks with an implicit batch dimension, this volume includes the batch dimension with its length set
    //! to the maximum batch size. For networks with all explicit dimensions and with wildcard dimensions, the volume
    //! is based on the maxima specified by an IOptimizationProfile.Dimensions are normally positive integers. The
    //! exception is that in networks with all explicit dimensions, -1 can be used as a wildcard for a dimension to
    //! be specified at runtime. Input tensors with such a wildcard must have a corresponding entry in the
    //! IOptimizationProfiles indicating the permitted extrema, and the input dimensions must be set by
    //! IExecutionContext::setBindingDimensions. Different IExecutionContext instances can have different dimensions.
    //! Wildcard dimensions are only supported for EngineCapability::kDEFAULT with DeviceType::kGPU. They are not
    //! supported in safety contexts or on the DLA.
    //!
    //! \param name The name of the tensor.
    //! \param type The type of the data held in the tensor.
    //! \param dimensions The dimensions of the tensor.
    //!
    //! \warning It is an error to specify a wildcard value on a dimension that is determined by trained parameters.
    //!
    //! \see ITensor
    //!
    //! \return The new tensor or nullptr if there is an error.
    //!
    virtual ITensor* addInput(const char* name, DataType type, Dims dimensions) TRTNOEXCEPT = 0;

    //!
    //! \brief Mark a tensor as a network output.
    //!
    //! \param tensor The tensor to mark as an output tensor.
    //!
    virtual void markOutput(ITensor& tensor) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a convolution layer to the network.
    //!
    //! \param input The input tensor to the convolution.
    //! \param nbOutputMaps The number of output feature maps for the convolution.
    //! \param kernelSize The HW-dimensions of the convolution kernel.
    //! \param kernelWeights The kernel weights for the convolution.
    //! \param biasWeights The optional bias weights for the convolution.
    //!
    //! \see IConvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new convolution layer, or nullptr if it could not be created.
    //!
    virtual IConvolutionLayer* addConvolution(ITensor& input, int nbOutputMaps, DimsHW kernelSize,
        Weights kernelWeights, Weights biasWeights) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a fully connected layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputs The number of outputs of the layer.
    //! \param kernelWeights The kernel weights for the convolution.
    //! \param biasWeights The optional bias weights for the convolution.
    //!
    //! \see IFullyConnectedLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new fully connected layer, or nullptr if it could not be created.
    //!
    virtual IFullyConnectedLayer* addFullyConnected(
        ITensor& input, int nbOutputs, Weights kernelWeights, Weights biasWeights) TRTNOEXCEPT = 0;

    //!
    //! \brief Add an activation layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param type The type of activation function to apply.
    //!
    //! Note that the setAlpha() and setBeta() methods must be used on the
    //! output for activations that require these parameters.
    //!
    //! \see IActivationLayer ActivationType
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new activation layer, or nullptr if it could not be created.
    //!
    virtual IActivationLayer* addActivation(ITensor& input, ActivationType type) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a pooling layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param type The type of pooling to apply.
    //! \param windowSize The size of the pooling window.
    //!
    //! \see IPoolingLayer PoolingType
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new pooling layer, or nullptr if it could not be created.
    //!
    virtual IPoolingLayer* addPooling(ITensor& input, PoolingType type, DimsHW windowSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a LRN layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param window The size of the window.
    //! \param alpha The alpha value for the LRN computation.
    //! \param beta The beta value for the LRN computation.
    //! \param k The k value for the LRN computation.
    //!
    //! \see ILRNLayer
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new LRN layer, or nullptr if it could not be created.
    //!
    virtual ILRNLayer* addLRN(ITensor& input, int window, float alpha, float beta, float k) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a Scale layer to the network.
    //!
    //! \param input The input tensor to The layer. This tensor is required to have a minimum of 3 dimensions.
    //! \param mode The scaling mode.
    //! \param shift The shift value.
    //! \param scale The scale value.
    //! \param power The power value.
    //!
    //! If the weights are available, then the size of weights are dependent on the on the ScaleMode.
    //! For ::kUNIFORM, the number of weights is equal to 1.
    //! For ::kCHANNEL, the number of weights is equal to the channel dimension.
    //! For ::kELEMENTWISE, the number of weights is equal to the volume of the input.
    //!
    //! \see IScaleLayer
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new Scale layer, or nullptr if it could not be created.
    //!
    virtual IScaleLayer* addScale(ITensor& input, ScaleMode mode, Weights shift, Weights scale, Weights power) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a SoftMax layer to the network.
    //!
    //! \see ISoftMaxLayer
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new SoftMax layer, or nullptr if it could not be created.
    //!
    virtual ISoftMaxLayer* addSoftMax(ITensor& input) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a concatenation layer to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //!
    //! \see IConcatenationLayer
    //!
    //! \return The new concatenation layer, or nullptr if it could not be created.
    //!
    //! \warning All tensors must have the same dimensions for all dimensions except for channel.
    //!
    virtual IConcatenationLayer* addConcatenation(ITensor* const* inputs, int nbInputs) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a deconvolution layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputMaps The number of output feature maps.
    //! \param kernelSize The HW-dimensions of the deconvolution kernel.
    //! \param kernelWeights The kernel weights for the deconvolution.
    //! \param biasWeights The optional bias weights for the deconvolution.
    //!
    //! \see IDeconvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new deconvolution layer, or nullptr if it could not be created.
    //!
    virtual IDeconvolutionLayer* addDeconvolution(ITensor& input, int nbOutputMaps, DimsHW kernelSize,
        Weights kernelWeights, Weights biasWeights) TRTNOEXCEPT = 0;

    //!
    //! \brief Add an elementwise layer to the network.
    //!
    //! \param input1 The first input tensor to the layer.
    //! \param input2 The second input tensor to the layer.
    //! \param op The binary operation that the layer applies.
    //!
    //! The input tensors must have the same number of dimensions.
    //! For each dimension, their lengths must match, or one of them must be one.
    //! In the latter case, the tensor is broadcast along that axis.
    //!
    //! The output tensor has the same number of dimensions as the inputs.
    //! For each dimension, its length is the maximum of the lengths of the
    //! corresponding input dimension.
    //!
    //! \see IElementWiseLayer
    //! \warning For shape tensors, ElementWiseOperation::kPOW is not a valid op.
    //!
    //! \return The new elementwise layer, or nullptr if it could not be created.
    //!
    virtual IElementWiseLayer* addElementWise(ITensor& input1, ITensor& input2, ElementWiseOperation op) TRTNOEXCEPT = 0;

    //!
    //! \brief Add an \p layerCount deep RNN layer to the network with a
    //! sequence length of \p maxSeqLen and \p hiddenSize internal state per
    //! layer.
    //!
    //! \param inputs The input tensor to the layer.
    //! \param layerCount The number of layers in the RNN.
    //! \param hiddenSize The size of the internal hidden state for each layer.
    //! \param maxSeqLen The maximum length of the time sequence.
    //! \param op The type of RNN to execute.
    //! \param mode The input mode for the RNN.
    //! \param dir The direction to run the RNN.
    //! \param weights The weights for the weight matrix parameters of the RNN.
    //! \param bias The weights for the bias vectors parameters of the RNN.
    //!
    //! The input tensors must be of the type DataType::kFLOAT or DataType::kHALF.
    //!
    //! See IRNNLayer::setWeights() and IRNNLayer::setBias() for details on the required input
    //! format for \p weights and \p bias.
    //!
    //! The layout for the \p input tensor should be `{1, S_max, N, E}`, where:
    //!   - `S_max` is the maximum allowed sequence length (number of RNN iterations)
    //!   - `N` is the batch size
    //!   - `E` specifies the embedding length (unless ::kSKIP is set, in which case it should match
    //!     getHiddenSize()).
    //!
    //! The first output tensor is the output of the final RNN layer across all timesteps, with dimensions
    //! `{S_max, N, H}`:
    //!
    //!   - `S_max` is the maximum allowed sequence length (number of RNN iterations)
    //!   - `N` is the batch size
    //!   - `H` is an output hidden state (equal to getHiddenSize() or 2x getHiddenSize())
    //!
    //! The second tensor is the final hidden state of the RNN across all layers, and if the RNN
    //! is an LSTM (i.e. getOperation() is ::kLSTM), then the third tensor is the final cell
    //! state of the RNN across all layers.  Both the second and third output tensors have dimensions
    //! `{L, N, H}`:
    //!
    //!  - `L` is equal to getLayerCount() if getDirection is ::kUNIDIRECTION,
    //!     and 2*getLayerCount() if getDirection is ::kBIDIRECTION.  In the bi-directional
    //!     case, layer `l`'s final forward hidden state is stored in `L = 2*l`, and
    //!     final backward hidden state is stored in `L = 2*l + 1`.
    //!  - `N` is the batch size
    //!  - `H` is getHiddenSize().
    //!
    //! Note that in bidirectional RNNs, the full "hidden state" for a layer `l`
    //! is the concatenation of its forward hidden state and its backward hidden
    //! state, and its size is 2*H.
    //!
    //! \deprecated IRNNLayer is superseded by IRNNv2Layer. Use addRNNv2() instead.
    //!
    //! \see IRNNLayer
    //!
    //! \warning RNN inputs do not support wildcard dimensions or explicit batch size networks.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new RNN layer, or nullptr if it could not be created.
    //!
    TRT_DEPRECATED virtual IRNNLayer* addRNN(ITensor& inputs, int layerCount, std::size_t hiddenSize, int maxSeqLen,
        RNNOperation op, RNNInputMode mode, RNNDirection dir, Weights weights, Weights bias) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a plugin layer to the network.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginLayer
    //!
    //! \deprecated IPluginLayer is superseded by IPluginV2. use addPluginV2 instead.
    //!
    //! \warning Plugin inputs do not support wildcard dimensions or explicit batch size networks.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return the new plugin layer, or nullptr if it could not be created.
    //!
    TRT_DEPRECATED virtual IPluginLayer* addPlugin(
        ITensor* const* inputs, int nbInputs, IPlugin& plugin) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a unary layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The operation to apply.
    //!
    //! \see IUnaryLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new unary layer, or nullptr if it could not be created
    //!
    virtual IUnaryLayer* addUnary(ITensor& input, UnaryOperation operation) TRTNOEXCEPT = 0;

    //! \brief Add a padding layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param prePadding The padding to apply to the start of the tensor.
    //! \param postPadding The padding to apply to the end of the tensor.
    //!
    //! \see IPaddingLayer
    //!
    //! \return The new padding layer, or nullptr if it could not be created.
    //!
    virtual IPaddingLayer* addPadding(ITensor& input, DimsHW prePadding, DimsHW postPadding) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a shuffle layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IShuffleLayer
    //!
    //! \return The new shuffle layer, or nullptr if it could not be created.
    //!
    virtual IShuffleLayer* addShuffle(ITensor& input) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the pooling output dimensions formula.
    //!
    //! \param formula The formula from computing the pooling output dimensions. If null is passed, the default
    //! formula is used.
    //!
    //! The default formula in each dimension is (inputDim + padding * 2 - kernelSize) / stride + 1.
    //!
    //! \warning Custom output dimensions formulas are not supported with wildcard dimensions.
    //!
    //! \see IOutputDimensionsFormula getPoolingOutputDimensionsFormula()
    //!
    TRT_DEPRECATED virtual void setPoolingOutputDimensionsFormula(IOutputDimensionsFormula* formula) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the pooling output dimensions formula.
    //!
    //! \return The formula from computing the pooling output dimensions.
    //!
    //! \warning Custom output dimensions formulas are not supported with wildcard dimensions.
    //!
    //! \see IOutputDimensionsFormula setPoolingOutputDimensionsFormula()
    //!
    TRT_DEPRECATED virtual IOutputDimensionsFormula& getPoolingOutputDimensionsFormula() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the convolution output dimensions formula.
    //!
    //! \deprecated This method does not currently work reliably and will be removed in a future release.
    //!
    //! \param formula The formula from computing the convolution output dimensions. If null is passed, the default
    //! formula is used.
    //!
    //! The default formula in each dimension is (inputDim + padding * 2 - kernelSize) / stride + 1.
    //!
    //! \warning Custom output dimensions formulas are not supported with wildcard dimensions.
    //!
    //! \see IOutputDimensionsFormula getConvolutionOutputDimensionsFormula()
    //!
    TRT_DEPRECATED virtual void setConvolutionOutputDimensionsFormula(
        IOutputDimensionsFormula* formula) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the convolution output dimensions formula.
    //!
    //! \deprecated This method does not currently work reliably and will be removed in a future release.
    //!
    //! \return The formula from computing the convolution output dimensions.
    //!
    //! \warning Custom output dimensions formulas are not supported with wildcard dimensions.
    //!
    //! \see IOutputDimensionsFormula setConvolutionOutputDimensionsFormula()
    //!
    TRT_DEPRECATED virtual IOutputDimensionsFormula& getConvolutionOutputDimensionsFormula() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the deconvolution output dimensions formula.
    //!
    //! \deprecated This method does not currently work reliably and will be removed in a future release.
    //!
    //! \param formula The formula from computing the deconvolution output dimensions. If null is passed, the default!
    //! formula is used.
    //!
    //! The default formula in each dimension is (inputDim - 1) * stride + kernelSize - 2 * padding.
    //!
    //! \warning Custom output dimensions formulas are not supported with wildcard dimensions.
    //!
    //! \see IOutputDimensionsFormula getDevonvolutionOutputDimensionsFormula()
    //!
    TRT_DEPRECATED virtual void setDeconvolutionOutputDimensionsFormula(
        IOutputDimensionsFormula* formula) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the deconvolution output dimensions formula.
    //!
    //! \return The formula from computing the deconvolution output dimensions.
    //!
    //! \deprecated This method does not currently work reliably and will be removed in a future release.
    //!
    //! \warning Custom output dimensions formulas are not supported with wildcard dimensions.
    //!
    //! \see IOutputDimensionsFormula setDeconvolutionOutputDimensionsFormula()
    //!
    TRT_DEPRECATED virtual IOutputDimensionsFormula& getDeconvolutionOutputDimensionsFormula() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the number of layers in the network.
    //!
    //! \return The number of layers in the network.
    //!
    //! \see getLayer()
    //!
    virtual int getNbLayers() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the layer specified by the given index.
    //!
    //! \param index The index of the layer.
    //!
    //! \return The layer, or nullptr if the index is out of range.
    //!
    //! \see getNbLayers()
    //!
    virtual ILayer* getLayer(int index) const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the number of inputs in the network.
    //!
    //! \return The number of inputs in the network.
    //!
    //! \see getInput()
    //!
    virtual int getNbInputs() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the input tensor specified by the given index.
    //!
    //! \param index The index of the input tensor.
    //!
    //! \return The input tensor, or nullptr if the index is out of range.
    //!
    //! \see getNbInputs()
    //!
    virtual ITensor* getInput(int index) const TRTNOEXCEPT = 0; // adding inputs invalidates indexing here

    //!
    //! \brief Get the number of outputs in the network.
    //!
    //! The outputs include those marked by markOutput or markOutputForShapes.
    //!
    //! \return The number of outputs in the network.
    //!
    //! \see getOutput()
    //!
    virtual int getNbOutputs() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the output tensor specified by the given index.
    //!
    //! \param index The index of the output tensor.
    //!
    //! \return The output tensor, or nullptr if the index is out of range.
    //!
    //! \see getNbOutputs()
    //!
    virtual ITensor* getOutput(int index) const TRTNOEXCEPT = 0; // adding outputs invalidates indexing here

    //!
    //! \brief Destroy this INetworkDefinition object.
    //!
    virtual void destroy() TRTNOEXCEPT = 0;

protected:
    virtual ~INetworkDefinition() {}

public:
    //!
    //! \brief Add a reduce layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param operation The reduction operation to perform.
    //! \param reduceAxes The reduction dimensions.
    //!        Bit 0 of the uint32_t type corresponds to the non-batch dimension 0 boolean and so on.
    //!        If a bit is set, then the corresponding dimension will be reduced.
    //!        Let's say we have an NCHW tensor as input (three non-batch dimensions).
    //!        Bit 0 corresponds to the C dimension boolean.
    //!        Bit 1 corresponds to the H dimension boolean.
    //!        Bit 2 corresponds to the W dimension boolean.
    //!        Note that reduction is not permitted over the batch size dimension.
    //!        When network has explicit batch mode enabled, dimensions 0 is the batch dimension.
    //! \param keepDimensions The boolean that specifies whether or not to keep the reduced dimensions in the
    //! output of the layer.
    //!
    //! \see IReduceLayer
    //!
    //! \warning If input is a shape tensor, ReduceOperation::kAVG is unsupported.
    //!
    //! \return The new reduce layer, or nullptr if it could not be created.
    //!
    virtual IReduceLayer* addReduce(ITensor& input, ReduceOperation operation, uint32_t reduceAxes, bool keepDimensions) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a TopK layer to the network.
    //!
    //! The TopK layer has two outputs of the same dimensions. The first contains data values,
    //! the second contains index positions for the values. Output values are sorted, largest first
    //! for operation kMAX and smallest first for operation kMIN.
    //!
    //! Currently only values of K up to 1024 are supported.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \param op Operation to perform.
    //!
    //! \param k Number of elements to keep.
    //!
    //! \param reduceAxes The reduction dimensions.
    //!        Bit 0 of the uint32_t type corresponds to the non-batch dimension 0 boolean and so on.
    //!        If a bit is set, then the corresponding dimension will be reduced.
    //!        Let's say we have an NCHW tensor as input (three non-batch dimensions).
    //!        Bit 0 corresponds to the C dimension boolean.
    //!        Bit 1 corresponds to the H dimension boolean.
    //!        Bit 2 corresponds to the W dimension boolean.
    //!        Note that TopK reduction is currently only permitted over one dimension.
    //!        When network has explicit batch mode enabled, dimensions 0 is the batch dimension.
    //!
    //! \see ITopKLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new TopK layer, or nullptr if it could not be created.
    //!
    virtual ITopKLayer* addTopK(ITensor& input, TopKOperation op, int k, uint32_t reduceAxes) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a gather layer to the network.
    //!
    //! \param data The tensor to gather values from.
    //! \param indices The tensor to get indices from to populate the output tensor.
    //! \param axis The non-batch dimension axis in the data tensor to gather on.
    //!
    //! \see IGatherLayer
    //!
    //! \return The new gather layer, or nullptr if it could not be created.
    //!
    virtual IGatherLayer* addGather(ITensor& data, ITensor& indices, int axis) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a RaggedSoftMax layer to the network.
    //!
    //! \param input The ZxS input tensor.
    //! \param bounds The Zx1 bounds tensor.
    //!
    //! \see IRaggedSoftMaxLayer
    //!
    //! \warning The bounds tensor cannot have the last dimension be the wildcard character.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new RaggedSoftMax layer, or nullptr if it could not be created.
    //!
    virtual IRaggedSoftMaxLayer* addRaggedSoftMax(ITensor& input, ITensor& bounds) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a MatrixMultiply layer to the network.
    //!
    //! \param input0 The first input tensor (commonly A).
    //! \param op0 The operation to apply to input0.
    //! \param input1 The second input tensor (commonly B).
    //! \param op1 The operation to apply to input1.
    //!
    //! \see IMatrixMultiplyLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new matrix multiply layer, or nullptr if it could not be created.
    //!
    virtual IMatrixMultiplyLayer* addMatrixMultiply(
        ITensor& input0, MatrixOperation op0, ITensor& input1, MatrixOperation op1) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a MatrixMultiply layer to the network.
    //!
    //! \param input0 The first input tensor (commonly A).
    //! \param transpose0 If true, op(input0)=transpose(input0), else op(input0)=input0.
    //! \param input1 The second input tensor (commonly B).
    //! \param transpose1 If true, op(input1)=transpose(input1), else op(input1)=input1.
    //!
    //! \see IMatrixMultiplyLayer
    //!
    //! \return The new matrix multiply layer, or nullptr if it could not be created.
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \deprecated This interface is superseded by the overload that replaces bool with MatrixOperation.
    //!
    TRT_DEPRECATED virtual IMatrixMultiplyLayer* addMatrixMultiply(
        ITensor& input0, bool transpose0, ITensor& input1, bool transpose1) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a constant layer to the network.
    //!
    //! \param dimensions The dimensions of the constant.
    //! \param weights The constant value, represented as weights.
    //!
    //! \see IConstantLayer
    //!
    //! \return The new constant layer, or nullptr if it could not be created.
    //!
    //! If weights.type is DataType::kINT32, the output is a tensor of 32-bit indices.
    //! Otherwise the output is a tensor of real values and the output type will be
    //! follow TensorRT's normal precision rules.
    //!
    //! If tensors in the network have an implicit batch dimension, the constant
    //! is broadcast over that dimension.
    //!
    //! If a wildcard dimension is used, the volume of the runtime dimensions must equal
    //! the number of weights specified.
    //!
    virtual IConstantLayer* addConstant(Dims dimensions, Weights weights) TRTNOEXCEPT = 0;

    //!
    //! \brief Add an \p layerCount deep RNN layer to the network with \p hiddenSize internal states that can
    //! take a batch with fixed or variable sequence lengths.
    //!
    //! \param input The input tensor to the layer (see below).
    //! \param layerCount The number of layers in the RNN.
    //! \param hiddenSize Size of the internal hidden state for each layer.
    //! \param maxSeqLen Maximum sequence length for the input.
    //! \param op The type of RNN to execute.
    //!
    //! By default, the layer is configured with RNNDirection::kUNIDIRECTION and RNNInputMode::kLINEAR.
    //! To change these settings, use IRNNv2Layer::setDirection() and IRNNv2Layer::setInputMode().
    //!
    //! %Weights and biases for the added layer should be set using
    //! IRNNv2Layer::setWeightsForGate() and IRNNv2Layer::setBiasForGate() prior
    //! to building an engine using this network.
    //!
    //! The input tensors must be of the type DataType::kFLOAT or DataType::kHALF.
    //! The layout of the weights is row major and must be the same datatype as the input tensor.
    //! \p weights contain 8 matrices and \p bias contains 8 vectors.
    //!
    //! See IRNNv2Layer::setWeightsForGate() and IRNNv2Layer::setBiasForGate() for details on the required input
    //! format for \p weights and \p bias.
    //!
    //! The \p input ITensor should contain zero or more index dimensions `{N1, ..., Np}`, followed by
    //! two dimensions, defined as follows:
    //!   - `S_max` is the maximum allowed sequence length (number of RNN iterations)
    //!   - `E` specifies the embedding length (unless ::kSKIP is set, in which case it should match
    //!     getHiddenSize()).
    //!
    //! By default, all sequences in the input are assumed to be size \p maxSeqLen.  To provide explicit sequence
    //! lengths for each input sequence in the batch, use IRNNv2Layer::setSequenceLengths().
    //!
    //! The RNN layer outputs up to three tensors.
    //!
    //! The first output tensor is the output of the final RNN layer across all timesteps, with dimensions
    //! `{N1, ..., Np, S_max, H}`:
    //!
    //!   - `N1..Np` are the index dimensions specified by the input tensor
    //!   - `S_max` is the maximum allowed sequence length (number of RNN iterations)
    //!   - `H` is an output hidden state (equal to getHiddenSize() or 2x getHiddenSize())
    //!
    //! The second tensor is the final hidden state of the RNN across all layers, and if the RNN
    //! is an LSTM (i.e. getOperation() is ::kLSTM), then the third tensor is the final cell state
    //! of the RNN across all layers.  Both the second and third output tensors have dimensions
    //! `{N1, ..., Np, L, H}`:
    //!
    //!  - `N1..Np` are the index dimensions specified by the input tensor
    //!  - `L` is the number of layers in the RNN, equal to getLayerCount() if getDirection is ::kUNIDIRECTION,
    //!     and 2x getLayerCount() if getDirection is ::kBIDIRECTION. In the bi-directional
    //!     case, layer `l`'s final forward hidden state is stored in `L = 2*l`, and
    //!     final backward hidden state is stored in `L= 2*l + 1`.
    //!  - `H` is the hidden state for each layer, equal to getHiddenSize().
    //!
    //! \see IRNNv2Layer
    //!
    //! \warning RNN inputs do not support wildcard dimensions or explicit batch size networks.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new RNN layer, or nullptr if it could not be created.
    //!
    virtual IRNNv2Layer* addRNNv2(
        ITensor& input, int32_t layerCount, int32_t hiddenSize, int32_t maxSeqLen, RNNOperation op) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a plugin layer to the network using an IPluginExt interface.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginLayer
    //!
    //! \deprecated IPluginLayer is superseded by IPluginV2. use addPluginV2 instead.
    //!
    //! \warning Plugin inputs do not support wildcard dimensions or explicit batch size networks.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new plugin layer, or nullptr if it could not be created.
    //!
    TRT_DEPRECATED virtual IPluginLayer* addPluginExt(
        ITensor* const* inputs, int nbInputs, IPluginExt& plugin) TRTNOEXCEPT = 0;

    //!
    //! \brief Add an identity layer.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IIdentityLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new identity layer, or nullptr if it could not be created.
    //!
    virtual IIdentityLayer* addIdentity(ITensor& input) TRTNOEXCEPT = 0;

    //!
    //! \brief remove a tensor from the network definition.
    //!
    //! \param tensor the tensor to remove
    //!
    //! It is illegal to remove a tensor that is the input or output of a layer.
    //! if this method is called with such a tensor, a warning will be emitted on the log
    //! and the call will be ignored. Its intended use is to remove detached tensors after
    //! e.g. concatenating two networks with Layer::setInput().
    //!
    virtual void removeTensor(ITensor& tensor) TRTNOEXCEPT = 0;

    //!
    //! \brief unmark a tensor as a network output.
    //!
    //! \param tensor The tensor to unmark as an output tensor.
    //!
    //! see markOutput()
    //!
    virtual void unmarkOutput(ITensor& tensor) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a plugin layer to the network using the IPluginV2 interface.
    //!
    //! \param inputs The input tensors to the layer.
    //! \param nbInputs The number of input tensors.
    //! \param plugin The layer plugin.
    //!
    //! \see IPluginV2Layer
    //!
    //! \warning Dimension wildcard are only supported with IPluginV2DynamicExt or IPluginV2IOExt plugins.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new plugin layer, or nullptr if it could not be created.
    //!
    virtual IPluginV2Layer* addPluginV2(ITensor* const* inputs, int nbInputs, IPluginV2& plugin) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a slice layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param start The start offset
    //! \param size The output dimension
    //! \param stride The slicing stride
    //!
    //! Positive, negative, zero stride values, and combinations of them in different dimensions are allowed.
    //!
    //! \see ISliceLayer
    //!
    //! \return The new slice layer, or nullptr if it could not be created.
    //!
    virtual ISliceLayer* addSlice(ITensor& input, Dims start, Dims size, Dims stride) TRTNOEXCEPT = 0;

    //!
    //! \brief Sets the name of the network.
    //!
    //! \param name The name to assign to this network.
    //!
    //! Set the name of the network so that it can be associated with a built
    //! engine. The \p name must be a zero delimited C-style string of length
    //! no greater than 128 characters. TensorRT makes no use of this string
    //! except storing it as part of the engine so that it may be retrieved at
    //! runtime. A name unique to the builder will be generated by default.
    //!
    //! This method copies the name string.
    //!
    //! \see INetworkDefinition::getName(), ISafeCudaEngine::getName()
    //!
    //! \return none
    //!
    virtual void setName(const char* name) TRTNOEXCEPT = 0;

    //!
    //! \brief Returns the name associated with the network.
    //!
    //! The memory pointed to by getName() is owned by the INetworkDefinition object.
    //!
    //! \see INetworkDefinition::setName()
    //!
    //! \return A zero delimited C-style string representing the name of the network.
    //!
    virtual const char* getName() const TRTNOEXCEPT = 0;

    //!
    //! \brief Add a shape layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IShapeLayer
    //!
    //! \warning addShape is only supported when hasImplicitBatchDimensions is false.
    //!
    //! \warning input to addShape cannot contain wildcard dimension values.
    //!
    //! \return The new shape layer, or nullptr if it could not be created.
    //!
    virtual IShapeLayer* addShape(ITensor& input) TRTNOEXCEPT = 0;

    //!
    //! \brief True if tensors have implicit batch dimension.
    //!
    //! \return True if tensors have implicit batch dimension, false otherwise.
    //!
    //! This is a network-wide property.  Either all tensors in the network
    //! have an implicit batch dimension or none of them do.
    //!
    //! hasImplicitBatchDimension() is true if and only if this INetworkDefinition
    //! was created with createNetwork() or createNetworkV2(true).
    //!
    //! \see createNetworkV2
    //!
    virtual bool hasImplicitBatchDimension() const TRTNOEXCEPT = 0;

    //!
    //! \brief Enable tensor's value to be computed by IExecutionContext::getShapeBinding.
    //!
    //! \return True if successful, false if tensor is already marked as an output.
    //!
    //! The tensor must be of type DataType::kINT32 and have no more than one dimension.
    //!
    //! \warning input to markOutputForShapes cannot contain wildcard dimension values.
    //!
    //! \see isShapeBinding(), getShapeBinding()
    //!
    virtual bool markOutputForShapes(ITensor& tensor) TRTNOEXCEPT = 0;

    //!
    //! \brief Undo markOutputForShapes.
    //!
    //! \warning inputs to addShape cannot contain wildcard dimension values.
    //!
    //! \return True if successful, false if tensor is not marked as an output.
    //!
    virtual bool unmarkOutputForShapes(ITensor& tensor) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a parametric ReLU layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param slope The slope tensor to the layer. This tensor should be unidirectionally broadcastable
    //!        to the input tensor.
    //!
    //! \see IParametricReLULayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new parametric ReLU layer, or nullptr if it could not be created.
    //!
    virtual IParametricReLULayer* addParametricReLU(ITensor& input, ITensor& slope) noexcept = 0;

    //!
    //! \brief Add a multi-dimension convolution layer to the network.
    //!
    //! \param input The input tensor to the convolution.
    //! \param nbOutputMaps The number of output feature maps for the convolution.
    //! \param kernelSize The multi-dimensions of the convolution kernel.
    //! \param kernelWeights The kernel weights for the convolution.
    //! \param biasWeights The optional bias weights for the convolution.
    //!
    //! \see IConvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new convolution layer, or nullptr if it could not be created.
    //!
    virtual IConvolutionLayer* addConvolutionNd(
        ITensor& input, int nbOutputMaps, Dims kernelSize, Weights kernelWeights, Weights biasWeights) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a multi-dimension pooling layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param type The type of pooling to apply.
    //! \param windowSize The size of the pooling window.
    //!
    //! \see IPoolingLayer PoolingType
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new pooling layer, or nullptr if it could not be created.
    //!
    virtual IPoolingLayer* addPoolingNd(ITensor& input, PoolingType type, Dims windowSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Add a multi-dimension deconvolution layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //! \param nbOutputMaps The number of output feature maps.
    //! \param kernelSize The multi-dimensions of the deconvolution kernel.
    //! \param kernelWeights The kernel weights for the deconvolution.
    //! \param biasWeights The optional bias weights for the deconvolution.
    //!
    //! \see IDeconvolutionLayer
    //!
    //! \warning It is an error to specify a wildcard value for the 'C' dimension of the input tensor.
    //! \warning Int32 tensors are not valid input tensors.
    //
    //! \return The new deconvolution layer, or nullptr if it could not be created.
    //!
    virtual IDeconvolutionLayer* addDeconvolutionNd(
        ITensor& input, int nbOutputMaps, Dims kernelSize, Weights kernelWeights, Weights biasWeights) TRTNOEXCEPT = 0;

    //! \brief Add a resize layer to the network.
    //!
    //! \param input The input tensor to the layer.
    //!
    //! \see IResizeLayer
    //!
    //! \warning Int32 tensors are not valid input tensors.
    //!
    //! \return The new resize layer, or nullptr if it could not be created.
    //!
    virtual IResizeLayer* addResize(ITensor& input) TRTNOEXCEPT = 0;
};

//!
//! enum CalibrationAlgoType
//!
//! \brief Version of calibration algorithm to use.
//!
enum class CalibrationAlgoType : int
{
    kLEGACY_CALIBRATION = 0,
    kENTROPY_CALIBRATION = 1,
    kENTROPY_CALIBRATION_2 = 2
};

template <>
constexpr inline int EnumMax<CalibrationAlgoType>()
{
    return 3;
} //!< Maximum number of elements in CalibrationAlgoType enum. \see DataType

//!
//! \class IInt8Calibrator
//!
//! \brief Application-implemented interface for calibration.
//!
//! Calibration is a step performed by the builder when deciding suitable scale factors for 8-bit inference.
//!
//! It must also provide a method for retrieving representative images which the calibration process can use to examine
//! the distribution of activations. It may optionally implement a method for caching the calibration result for reuse
//! on subsequent runs.
//!
class IInt8Calibrator
{
public:
    //!
    //! \brief Get the batch size used for calibration batches.
    //!
    //! \return The batch size.
    //!
    virtual int getBatchSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get a batch of input for calibration.
    //!
    //! The batch size of the input must match the batch size returned by getBatchSize().
    //!
    //! \param bindings An array of pointers to device memory that must be updated to point to device memory
    //! containing each network input data.
    //! \param names The names of the network input for each pointer in the binding array.
    //! \param nbBindings The number of pointers in the bindings array.
    //! \return False if there are no more batches for calibration.
    //!
    //! \see getBatchSize()
    //!
    virtual bool getBatch(void* bindings[], const char* names[], int nbBindings) TRTNOEXCEPT = 0;

    //!
    //! \brief Load a calibration cache.
    //!
    //! Calibration is potentially expensive, so it can be useful to generate the calibration data once, then use it on
    //! subsequent builds of the network. The cache includes the regression cutoff and quantile values used to generate
    //! it, and will not be used if these do not batch the settings of the current calibrator. However, the network
    //! should also be recalibrated if its structure changes, or the input data set changes, and it is the
    //! responsibility of the application to ensure this.
    //!
    //! \param length The length of the cached data, that should be set by the called function. If there is no data,
    //! this should be zero.
    //!
    //! \return A pointer to the cache, or nullptr if there is no data.
    //!
    virtual const void* readCalibrationCache(std::size_t& length) TRTNOEXCEPT = 0;

    //!
    //! \brief Save a calibration cache.
    //!
    //! \param ptr A pointer to the data to cache.
    //! \param length The length in bytes of the data to cache.
    //!
    //! \see readCalibrationCache()
    //!
    virtual void writeCalibrationCache(const void* ptr, std::size_t length) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the algorithm used by this calibrator.
    //!
    //! \return The algorithm used by the calibrator.
    //!
    virtual CalibrationAlgoType getAlgorithm() TRTNOEXCEPT = 0;

    virtual ~IInt8Calibrator() {}
};

//!
//! Entropy calibrator. This is the Legacy Entropy calibrator. It is less complicated than the legacy calibrator and
//! produces better results.
//!
class IInt8EntropyCalibrator : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the entropy calibrator.
    //!
    CalibrationAlgoType getAlgorithm() TRTNOEXCEPT override { return CalibrationAlgoType::kENTROPY_CALIBRATION; }

    virtual ~IInt8EntropyCalibrator() {}
};

//!
//! Entropy calibrator 2. This is the preferred calibrator. This is the required calibrator for DLA, as it supports per
//! activation tensor scaling.
//!
class IInt8EntropyCalibrator2 : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the entropy calibrator 2.
    //!
    CalibrationAlgoType getAlgorithm() TRTNOEXCEPT override { return CalibrationAlgoType::kENTROPY_CALIBRATION_2; }

    virtual ~IInt8EntropyCalibrator2() {}
};

//!
//! \deprecated Legacy calibrator left for backward compatibility with TensorRT 2.0.
//!
class TRT_DEPRECATED IInt8LegacyCalibrator : public IInt8Calibrator
{
public:
    //!
    //! Signal that this is the legacy calibrator.
    //!
    CalibrationAlgoType getAlgorithm() TRTNOEXCEPT override { return CalibrationAlgoType::kLEGACY_CALIBRATION; }

    //!
    //! \brief The quantile (between 0 and 1) that will be used to select the region maximum when the quantile method
    //! is in use.
    //!
    //! See the user guide for more details on how the quantile is used.
    //!
    virtual double getQuantile() const TRTNOEXCEPT = 0;

    //!
    //! \brief The fraction (between 0 and 1) of the maximum used to define the regression cutoff when using regression
    //! to determine the region maximum.
    //!
    //! See the user guide for more details on how the regression cutoff is used
    //!
    virtual double getRegressionCutoff() const TRTNOEXCEPT = 0;

    //!
    //! \brief Load a histogram.
    //!
    //! Histogram generation is potentially expensive, so it can be useful to generate the histograms once, then use
    //! them when exploring the space of calibrations. The histograms should be regenerated if the network structure
    //! changes, or the input data set changes, and it is the responsibility of the application to ensure this.
    //!
    //! \param length The length of the cached data, that should be set by the called function. If there is no data,
    //! this should be zero.
    //!
    //! \return A pointer to the cache, or nullptr if there is no data.
    //!
    virtual const void* readHistogramCache(std::size_t& length) TRTNOEXCEPT = 0;

    //!
    //! \brief Save a histogram cache.
    //!
    //! \param ptr A pointer to the data to cache.
    //! \param length The length in bytes of the data to cache.
    //!
    //! \see readHistogramCache()
    //!
    virtual void writeHistogramCache(const void* ptr, std::size_t length) TRTNOEXCEPT = 0;

    virtual ~IInt8LegacyCalibrator() {}
};

//!
//! \brief It is capable of representing one or more BuilderFlags by binary OR
//! operations, e.g., 1U << BuilderFlag::kFP16 | 1U << BuilderFlag::kDEBUG.
//!
//! \see INetworkConfig::getFlags(), ITensor::setFlags(),
//!
typedef uint32_t BuilderFlags;

//!
//! \enum BuilderFlags
//!
//! \brief List of valid modes that the builder can enable when creating an engine from a network definition.
//!
//! \see INetworkConfig::setFlag(), INetworkConfig::getFlag()
//!
enum class BuilderFlag : int
{
    kFP16           = 0,    //!< Enable FP16 layer selection.
    kINT8           = 1,    //!< Enable Int8 layer selection.
    kDEBUG          = 2,    //!< Enable debugging of layers via synchronizing after every layer.
    kGPU_FALLBACK   = 3,    //!< Enable layers marked to execute on GPU if layer cannot execute on DLA.
    kSTRICT_TYPES   = 4,    //!< Enables strict type constraints.
    kREFIT          = 5,    //!< Enable building a refittable engine.
};

template <>
constexpr inline int EnumMax<BuilderFlag>()
{
    return 6;
} //!< Maximum number of builder flags in BuilderFlag enum. \see BuilderFlag

//!
//! \class INetworkConfig
//!
//! \brief Holds properties for configuring a network for an engine. \see BuilderFlags
//!
class INetworkConfig
{
public:
    //!
    //! \brief Set the number of minimization iterations used when timing layers.
    //!
    //! When timing layers, the builder minimizes over a set of average times for layer execution. This parameter
    //! controls the number of iterations used in minimization. The builder may sometimes run layers for more
    //! iterations to improve timing accuracy if this parameter is set to a small value and the runtime of the
    //! layer is short.
    //!
    //! \see getMinTimingIterations()
    //!
    virtual void setMinTimingIterations(int minTiming) TRTNOEXCEPT = 0;

    //!
    //! \brief Query the number of minimization iterations.
    //!
    //! \see setMinTimingIterations()
    //!
    virtual int getMinTimingIterations() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the number of averaging iterations used when timing layers.
    //!
    //! When timing layers, the builder minimizes over a set of average times for layer execution. This parameter
    //! controls the number of iterations used in averaging.
    //!
    //! \see getAvgTimingIterations()
    //!
    virtual void setAvgTimingIterations(int avgTiming) TRTNOEXCEPT = 0;

    //!
    //! \brief Query the number of averaging iterations.
    //!
    //! \see setAvgTimingIterations()
    //!
    virtual int getAvgTimingIterations() const TRTNOEXCEPT = 0;

    //!
    //! \brief Configure the builder to target specified EngineCapability flow.
    //!
    //! The flow means a sequence of API calls that allow an application to set up a runtime, engine,
    //! and execution context in order to run inference.
    //!
    //! The supported flows are specified in the EngineCapability enum.
    //!
    virtual void setEngineCapability(EngineCapability capability) TRTNOEXCEPT = 0;

    //!
    //! \brief Query EngineCapability flow configured for the builder.
    //!
    //! By default it returns EngineCapability::kDEFAULT.
    //!
    //! \see setEngineCapability()
    //!
    virtual EngineCapability getEngineCapability() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set Int8 Calibration interface.
    //!
    //! The calibrator is to minimize the information loss during the INT8 quantization process.
    //!
    virtual void setInt8Calibrator(IInt8Calibrator* calibrator) TRTNOEXCEPT = 0;

    //!
    //! \brief Get Int8 Calibration interface.
    //!
    virtual IInt8Calibrator* getInt8Calibrator() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the maximum workspace size.
    //!
    //! \param workspaceSize The maximum GPU temporary memory which the engine can use at execution time.
    //!
    //! \see getMaxWorkspaceSize()
    //!
    virtual void setMaxWorkspaceSize(std::size_t workspaceSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the maximum workspace size.
    //!
    //! \return The maximum workspace size.
    //!
    //! \see setMaxWorkspaceSize()
    //!
    virtual std::size_t getMaxWorkspaceSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the build mode flags to turn on builder options for this network.
    //!
    //! The flags are listed in the BuilderFlags enum.
    //! The flags set configuration options to build the network.
    //!
    //! \param builderFlags The build option for an engine.
    //!
    //! \note This function will override the previous set flags, rather than bitwise adding the new flag.
    //!
    //! \see getFlags()
    //!
    virtual void setFlags(BuilderFlags builderFlags) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the set of build mode flags for this network config. Defaults to BuildMode::kDEFAULT.
    //!
    //! \return The build options as a bitmask.
    //!
    //! \see setFlags()
    //!
    virtual BuilderFlags getFlags() const TRTNOEXCEPT = 0;

    //!
    //! \brief clear a single build mode flag.
    //!
    //! clears the builder flag from the set of enabled flags.
    //!
    //! \see setFlags
    //!
    virtual void clearFlag(BuilderFlag builderFlag) TRTNOEXCEPT = 0;

    //!
    //! \brief Set a single build mode flag.
    //!
    //! Sets the build mode flags on top of the flags already specified.
    //!
    //! \see setFlags
    //!
    virtual void setFlag(BuilderFlag builderFlag) TRTNOEXCEPT = 0;

    //!
    //! \brief returns true if the build mode flag is set
    //!
    //! Check if a build mode flag is set.
    //!
    //! \see getFlags()
    //!
    //! \return True if flag is set, false if unset.
    //!
    virtual bool getFlag(BuilderFlag builderFlag) const TRTNOEXCEPT = 0;


    //!
    //! \brief Set the device that this layer must execute on.
    //! \param DeviceType that this layer must execute on.
    //! If DeviceType is not set or is reset, TensorRT will use the default DeviceType set in the builder.
    //!
    //! \note The device type for a layer must be compatible with the safety flow (if specified).
    //! For example a layer cannot be marked for DLA execution while the builder is configured for kSAFE_GPU.
    //!
    //! \see getDeviceType()
    //!
    virtual void setDeviceType(const ILayer* layer, DeviceType deviceType) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the device that this layer executes on.
    //! \return Returns DeviceType of the layer.
    //!
    virtual DeviceType getDeviceType(const ILayer* layer) const TRTNOEXCEPT = 0;

    //!
    //! \brief whether the DeviceType has been explicitly set for this layer
    //! \return true if device type is not default
    //! \see setDeviceType() getDeviceType() resetDeviceType()
    //!
    virtual bool isDeviceTypeSet(const ILayer* layer) const TRTNOEXCEPT = 0;

    //!
    //! \brief reset the DeviceType for this layer
    //!
    //! \see setDeviceType() getDeviceType() isDeviceTypeSet()
    //!
    virtual void resetDeviceType(const ILayer* layer) TRTNOEXCEPT = 0;

    //!
    //! \brief Checks if a layer can run on DLA.
    //! \return status true if the layer can on DLA else returns false.
    //!
    virtual bool canRunOnDLA(const ILayer* layer) const TRTNOEXCEPT = 0;

    //!
    //! \brief Sets the DLA core used by the network.
    //! \param dlaCore The DLA core to execute the engine on (0 to N-1). Default value is 0.
    //!
    //! It can be used to specify which DLA core to use via indexing, if multiple DLA cores are available.
    //!
    //! \see IRuntime::setDLACore() getDLACore()
    //!
    virtual void setDLACore(int dlaCore) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the DLA core that the engine executes on.
    //! \return If setDLACore is called, returns DLA core from 0 to N-1, else returns 0.
    //!
    virtual int getDLACore() const TRTNOEXCEPT = 0;

    //!
    //! \brief Sets the default DeviceType to be used by the builder. It ensures that all the layers that can run on
    //! this device will run on it, unless setDeviceType is used to override the default DeviceType for a layer.
    //! \see getDefaultDeviceType()
    //!
    virtual void setDefaultDeviceType(DeviceType deviceType) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the default DeviceType which was set by setDefaultDeviceType.
    //!
    virtual DeviceType getDefaultDeviceType() const TRTNOEXCEPT = 0;

    //!
    //! \brief Resets the network configuration to defaults.
    //!
    //! When initializing a network config object, we can call this function.
    //!
    virtual void reset() TRTNOEXCEPT = 0;

    //!
    //! \brief De-allocates any internally allocated memory.
    //!
    //! When destroying a network config object, we can call this function.
    //!
    virtual void destroy() TRTNOEXCEPT = 0;

    //!
    //! \brief Set the cudaStream that is used to profile this network.
    //!
    //! \param stream The cuda stream used for profiling by the builder.
    //!
    //! \see getProfileStream()
    //!
    virtual void setProfileStream(const cudaStream_t stream) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the cudaStream that is used to profile this network.
    //!
    //! \return The cuda stream used for profiling by the builder.
    //!
    //! \see setProfileStream()
    //!
    virtual cudaStream_t getProfileStream() const TRTNOEXCEPT = 0;

    //!
    //! \brief Add an optimization profile.
    //!
    //! This function must be called at least once if the network has dynamic or shape input tensors.
    //! This function may be called at most once when building a refittable engine, as more than
    //! a single optimization profile are not supported for refittable engines.
    //!
    //! \param profile The new optimization profile, which must satisfy profile->isValid() == true
    //! \return The index of the optimization profile (starting from 0) if the input is valid, or -1 if the input is
    //!         not valid.
    //!
    virtual int addOptimizationProfile(const IOptimizationProfile* profile) noexcept = 0;

    //!
    //! \brief Get number of optimization profiles
    //!
    //! This is one higher than the index of the last optimization profile that has be defined (or
    //! zero, if none has been defined yet).
    //!
    virtual int getNbOptimizationProfiles() const noexcept = 0;

    //!
    //! \brief Enable pointwise fusion.
    //!
    //! \param mode True to enable pointwise fusion else disable. Default value is true.
    //!
    //! \see getEnablePointWiseFusion()
    //!
    virtual void setEnablePointWiseFusion(bool mode) TRTNOEXCEPT = 0;
    
    //!
    //! \brief Check if pointwise fusion is enabled.
    //!
    //! \return Status true if pointwise fusion is enabled else returns false.
    //!
    //! \see setEnablePointWiseFusion()
    //!
    virtual bool getEnablePointWiseFusion() const TRTNOEXCEPT = 0;
protected:
    virtual ~INetworkConfig() {}
};

//!
//! \class IBuilder
//!
//! \brief Builds an engine from a network definition.
//!
//! \warning Do not inherit from this class, as doing so will break forward-compatibility of the API and ABI.
//!
class IBuilder
{
public:
    //!
    //! \brief Create a network definition object where all tensors have an implicit batch dimension.
    //!
    //! This method is equivalent to createNetworkV2(true), and retained for compatibility
    //! with earlier version of TensorRT.  The network does not support dynamic shapes or explicit batch sizes.
    //!
    //! \see INetworkDefinition, createNetworkV2
    //!
    //! \deprecated API will be removed in a future release, use IBuilder::createNetworkV2() instead.
    //!
    TRT_DEPRECATED virtual nvinfer1::INetworkDefinition* createNetwork() TRTNOEXCEPT = 0;

    //!
    //! \brief Set the maximum batch size.
    //!
    //! \param batchSize The maximum batch size which can be used at execution time, and also the batch size for which
    //! the engine will be optimized.
    //!
    //! \see getMaxBatchSize()
    //!
    virtual void setMaxBatchSize(int batchSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the maximum batch size.
    //!
    //! \return The maximum batch size.
    //!
    //! \see setMaxBatchSize()
    //! \see getMaxDLABatchSize()
    //!
    virtual int getMaxBatchSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the maximum workspace size.
    //!
    //! \param workspaceSize The maximum GPU temporary memory which the engine can use at execution time.
    //!
    //! \see getMaxWorkspaceSize()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setMaxWorkspaceSize instead.
    //!
    TRT_DEPRECATED virtual void setMaxWorkspaceSize(std::size_t workspaceSize) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the maximum workspace size.
    //!
    //! \return The maximum workspace size.
    //!
    //! \see setMaxWorkspaceSize()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getMaxWorkspaceSize instead.
    //!
    TRT_DEPRECATED virtual std::size_t getMaxWorkspaceSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set whether half2 mode is used.
    //!
    //! half2 mode is a paired-image mode that is significantly faster for batch sizes greater than one on platforms
    //! with fp16 support.
    //!
    //! \param mode Whether half2 mode is used.
    //!
    //! \see getHalf2Mode()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setFlag instead.
    //!
    TRT_DEPRECATED virtual void setHalf2Mode(bool mode) TRTNOEXCEPT = 0;

    //!
    //! \brief Query whether half2 mode is used.
    //!
    //! \see setHalf2Mode()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getFlag instead.
    //!
    TRT_DEPRECATED virtual bool getHalf2Mode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set whether the builder should use debug synchronization.
    //!
    //! If this flag is true, the builder will synchronize after timing each layer, and report the layer name. It can
    //! be useful when diagnosing issues at build time.
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setFlag instead.
    //!
    TRT_DEPRECATED virtual void setDebugSync(bool sync) TRTNOEXCEPT = 0;

    //!
    //! \brief Query whether the builder will use debug synchronization.
    //!
    //! \see setDebugSync()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getFlag instead.
    //!
    TRT_DEPRECATED virtual bool getDebugSync() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the number of minimization iterations used when timing layers.
    //!
    //! When timing layers, the builder minimizes over a set of average times for layer execution. This parameter
    //! controls the number of iterations used in minimization.
    //!
    //! \see getMinFindIterations()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setMinTimingIterations instead.
    //!
    TRT_DEPRECATED virtual void setMinFindIterations(int minFind) TRTNOEXCEPT = 0;

    //!
    //! \brief Query the number of minimization iterations.
    //!
    //! \see setMinFindIterations()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getMinTimingIterations instead.
    //!
    TRT_DEPRECATED virtual int getMinFindIterations() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the number of averaging iterations used when timing layers.
    //!
    //! When timing layers, the builder minimizes over a set of average times for layer execution. This parameter
    //! controls the number of iterations used in averaging.
    //!
    //! \see getAverageFindIterations()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setAvgTimingIterations instead.
    //!
    TRT_DEPRECATED virtual void setAverageFindIterations(int avgFind) TRTNOEXCEPT = 0;

    //!
    //! \brief Query the number of averaging iterations.
    //!
    //! \see setAverageFindIterations()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getAvgTimingIterations instead.
    //!
    TRT_DEPRECATED virtual int getAverageFindIterations() const TRTNOEXCEPT = 0;

    //!
    //! \brief Build a CUDA engine from a network definition.
    //!
    //! \see INetworkDefinition ICudaEngine
    //!
    //! \depercated API will be removed in a future release, use INetworkConfig::buildEngineWithConfig instead.
    //!
    TRT_DEPRECATED virtual nvinfer1::ICudaEngine* buildCudaEngine(
        nvinfer1::INetworkDefinition& network) TRTNOEXCEPT = 0;

    //!
    //! \brief Determine whether the platform has fast native fp16.
    //!
    virtual bool platformHasFastFp16() const TRTNOEXCEPT = 0;

    //!
    //! \brief Determine whether the platform has fast native int8.
    //!
    virtual bool platformHasFastInt8() const TRTNOEXCEPT = 0;

    //!
    //! \brief Destroy this object.
    //!
    virtual void destroy() TRTNOEXCEPT = 0;

    //!
    //! \brief Set whether or not quantized 8-bit kernels are permitted.
    //!
    //! During engine build int8 kernels will also be tried when this mode is enabled.
    //!
    //! \param mode Whether quantized 8-bit kernels are permitted.
    //!
    //! \see getInt8Mode()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setFlag instead.
    //!
    TRT_DEPRECATED virtual void setInt8Mode(bool mode) TRTNOEXCEPT = 0;

    //!
    //! \brief Query whether Int8 mode is used.
    //!
    //! \see setInt8Mode()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getFlag instead.
    //!
    TRT_DEPRECATED virtual bool getInt8Mode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set Int8 Calibration interface.
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setInt8Calibrator instead.
    //!
    TRT_DEPRECATED virtual void setInt8Calibrator(IInt8Calibrator* calibrator) TRTNOEXCEPT = 0;

    //!
    //! \brief Set the device that this layer must execute on.
    //! \param DeviceType that this layer must execute on.
    //! If DeviceType is not set or is reset, TensorRT will use the default DeviceType set in the builder.
    //!
    //! \note The device type for a layer must be compatible with the safety flow (if specified).
    //! For example a layer cannot be marked for DLA execution while the builder is configured for kSAFE_GPU.
    //!
    //! \see getDeviceType()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setDeviceType instead.
    //!
    TRT_DEPRECATED virtual void setDeviceType(ILayer* layer, DeviceType deviceType) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the device that this layer executes on.
    //! \return Returns DeviceType of the layer.
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getDeviceType instead.
    //!
    TRT_DEPRECATED virtual DeviceType getDeviceType(const ILayer* layer) const TRTNOEXCEPT = 0;

    //!
    //! \brief whether the DeviceType has been explicitly set for this layer
    //! \return whether the DeviceType has been explicitly set
    //! \see setDeviceType() getDeviceType() resetDeviceType()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::isDeviceTypeSet instead.
    //!
    TRT_DEPRECATED virtual bool isDeviceTypeSet(const ILayer* layer) const TRTNOEXCEPT = 0;

    //!
    //! \brief reset the DeviceType for this layer
    //!
    //! \see setDeviceType() getDeviceType() isDeviceTypeSet()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::resetDeviceType instead.
    //!
    TRT_DEPRECATED virtual void resetDeviceType(ILayer* layer) TRTNOEXCEPT = 0;

    //!
    //! \brief Checks if a layer can run on DLA.
    //! \return status true if the layer can on DLA else returns false.
    //!
    TRT_DEPRECATED virtual bool canRunOnDLA(const ILayer* layer) const TRTNOEXCEPT = 0;

    //!
    //! \brief Sets the default DeviceType to be used by the builder. It ensures that all the layers that can run on
    //! this device will run on it, unless setDeviceType is used to override the default DeviceType for a layer.
    //! \see getDefaultDeviceType()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setDefaultDeviceType instead.
    //!
    TRT_DEPRECATED virtual void setDefaultDeviceType(DeviceType deviceType) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the default DeviceType which was set by setDefaultDeviceType.
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getDefaultDeviceType instead.
    //!
    TRT_DEPRECATED virtual DeviceType getDefaultDeviceType() const TRTNOEXCEPT = 0;

    //!
    //! \brief Get the maximum batch size DLA can support.
    //! For any tensor the total volume of index dimensions combined(dimensions other than CHW) with the requested
    //! batch size should not exceed the value returned by this function.
    //!
    //! \warning getMaxDLABatchSize does not work with dynamic shapes.
    //!
    virtual int getMaxDLABatchSize() const TRTNOEXCEPT = 0;

    //!
    //! \brief Sets the builder to use GPU if a layer that was supposed to run on DLA can not run on DLA.
    //! \param Allows fallback if setFallBackMode is true else disables fallback option.
    //!
    //! \note GPU fallback may only be specified for non-safety modes. \see EngineCapability
    //! Simultaneously enabling GPU fallback and safety-restricted modes is disallowed.
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setFlag instead.
    //!
    TRT_DEPRECATED virtual void allowGPUFallback(bool setFallBackMode) TRTNOEXCEPT = 0;

    //!
    //! \brief Return the number of DLA engines available to this builder.
    //!
    virtual int getNbDLACores() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set the DLA core that the engine must execute on.
    //! \param dlaCore The DLA core to execute the engine on (0 to N-1, where N is the maximum number of DLA cores
    //! present on the device). Default value is 0.
    //! DLA Core is not a property of the engine that is preserved by serialization: when the engine is deserialized
    //! it will be associated with the DLA core which is configured for the runtime.
    //! \see IRuntime::setDLACore() getDLACore()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setDLACore instead.
    //!
    TRT_DEPRECATED virtual void setDLACore(int dlaCore) TRTNOEXCEPT = 0;

    //!
    //! \brief Get the DLA core that the engine executes on.
    //! \return If setDLACore is called, returns DLA core from 0 to N-1, else returns 0.
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getDLACore instead.
    //!
    TRT_DEPRECATED virtual int getDLACore() const TRTNOEXCEPT = 0;

    //!
    //! \brief Resets the builder state
    //!
    //! \deprecated API will be removed in a future release, use IBuilder::reset() instead.
    //!
    TRT_DEPRECATED virtual void reset(nvinfer1::INetworkDefinition& network) TRTNOEXCEPT = 0;

protected:
    virtual ~IBuilder()
    {
    }

public:
    //!
    //! \brief Set the GPU allocator.
    //! \param allocator Set the GPU allocator to be used by the builder. All GPU memory acquired will use this
    //! allocator. If NULL is passed, the default allocator will be used.
    //!
    //! Default: uses cudaMalloc/cudaFree.
    //!
    //! \note This allocator will be passed to any engines created via the builder; thus the lifetime of the allocator
    //! must span the lifetime of those engines as
    //! well as that of the builder. If nullptr is passed, the default allocator will be used.
    //!
    virtual void setGpuAllocator(IGpuAllocator* allocator) TRTNOEXCEPT = 0;

    //!
    //! \brief Set whether or not 16-bit kernels are permitted.
    //!
    //! During engine build fp16 kernels will also be tried when this mode is enabled.
    //!
    //! \param mode Whether 16-bit kernels are permitted.
    //!
    //! \see getFp16Mode()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setFlag instead.
    //!
    TRT_DEPRECATED virtual void setFp16Mode(bool mode) TRTNOEXCEPT = 0;

    //!
    //! \brief Query whether 16-bit kernels are permitted.
    //!
    //! \see setFp16Mode()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getFlag instead.
    //!
    TRT_DEPRECATED virtual bool getFp16Mode() const TRTNOEXCEPT = 0;

    //!
    //! \brief Set whether or not type constraints are strict.
    //!
    //! When strict type constraints are in use, TensorRT will always choose a layer implementation that conforms to the
    //! type constraints specified, if one exists. If this flag is not set, a higher-precision implementation may be
    //! chosen if it results in higher performance.
    //!
    //! If no conformant layer exists, TensorRT will choose a non-conformant layer if available regardless of the
    //! setting of this flag.
    //!
    //! See the developer guide for the definition of strictness.
    //!
    //! \param mode Whether type constraints are strict
    //!
    //! \see getStrictTypeConstraints()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setFlag instead.
    //!
    TRT_DEPRECATED virtual void setStrictTypeConstraints(bool mode) TRTNOEXCEPT = 0;

    //!
    //! \brief Query whether or not type constraints are strict.
    //!
    //! \see setStrictTypeConstraints()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getFlag instead.
    //!
    TRT_DEPRECATED virtual bool getStrictTypeConstraints() const TRTNOEXCEPT = 0;

    //!
    //! Set whether engines will be refittable.
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setFlag instead.
    //!
    TRT_DEPRECATED virtual void setRefittable(bool canRefit) TRTNOEXCEPT = 0;

    //!
    //! \brief Query whether or not engines will be refittable.
    //!
    //! \see getRefittable()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getFlag instead.
    //!
    TRT_DEPRECATED virtual bool getRefittable() const TRTNOEXCEPT = 0;

    //!
    //! \brief Configure the builder to target specified EngineCapability flow.
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::setEngineCapability instead.
    //!
    TRT_DEPRECATED virtual void setEngineCapability(EngineCapability capability) TRTNOEXCEPT = 0;

    //!
    //! \brief Query EngineCapability flow configured for the builder.
    //!
    //! \see setEngineCapability()
    //!
    //! \deprecated API will be removed in a future release, use INetworkConfig::getEngineCapability instead.
    //!
    TRT_DEPRECATED virtual EngineCapability getEngineCapability() const TRTNOEXCEPT = 0;

    //!
    //! \brief Create a network configuration object.
    //!
    //! \see INetworkConfig
    //!
    virtual nvinfer1::INetworkConfig* createNetworkConfig() TRTNOEXCEPT = 0;

    //!
    //! \brief Builds an engine for the given INetworkDefinition and given INetworkConfig.
    //!
    //! It enables the builder to build multiple engines based on the same network definition, but with different
    //! network configurations.
    //!
    virtual nvinfer1::ICudaEngine* buildEngineWithConfig(INetworkDefinition& network, INetworkConfig& config) TRTNOEXCEPT = 0;

    //! \brief Create a network definition object.
    //!
    //! \param implicitBatchDimension true if tensors have an implicit batch dimension.
    //!
    //! In TensorRT 5.1 and prior, tensors defined by the network always had an implicit batch dimension,
    //! and this dimension was specified at execution by a batchSize parameter.
    //! Use implicitBatchDimension=true for compatibility with those versions.
    //!
    //! Dynamic shape support requires implicitBatchDimension=false.
    //! With dynamic shapes, any of the input dimensions can vary at run-time,
    //! and there are no implicit dimensions in the network specification. This is specified by using the
    //! wildcard dimension value -1.
    //!
    //! \see INetworkDefinition, hasImplicitBatchDimension
    //!
    virtual nvinfer1::INetworkDefinition* createNetworkV2(bool implicitBatchDimension) TRTNOEXCEPT = 0;

    //! \brief Create a new optimization profile.
    //!
    //! If the network has any dynamic input tensors, the appropriate calls to setDimensions() must be made.
    //! Likewise, if there are any shape input tensors, the appropriate calls to setShapeValues() are required.
    //! The builder retains ownership of the created optimization profile and returns a raw pointer, i.e. the users
    //! must not attempt to delete the returned pointer.
    //!
    //! \see IOptimizationProfile
    //!
    virtual nvinfer1::IOptimizationProfile* createOptimizationProfile() noexcept = 0;

    //!
    //! \brief Set the ErrorRecorder for this interface
    //!
    //! Assigns the ErrorRecorder to this interface. The ErrorRecorder will track all errors during execution.
    //! This function will call incRefCount of the registered ErrorRecorder at least once. Setting
    //! recorder to nullptr unregisters the recorder with the interface, resulting in a call to decRefCount if
    //! a recorder has been registered.
    //!
    //! \param recorder The error recorder to register with this interface.
    //
    //! \see getErrorRecorder
    //!
    virtual void setErrorRecorder(IErrorRecorder* recorder) TRTNOEXCEPT = 0;

    //!
    //! \brief get the ErrorRecorder assigned to this interface.
    //!
    //! Retrieves the assigned error recorder object for the given class. A default error recorder does not exist,
    //! so a nullptr will be returned if setErrorRecorder has not been called.
    //!
    //! \return A pointer to the IErrorRecorder object that has been registered.
    //!
    //! \see setErrorRecorder
    //!
    virtual IErrorRecorder* getErrorRecorder() const TRTNOEXCEPT = 0;

    //!
    //! \brief Resets the builder state to default values.
    //!
    virtual void reset() TRTNOEXCEPT = 0;
};

} // namespace nvinfer1

extern "C" TENSORRTAPI void* createInferBuilder_INTERNAL(void* logger, int version); //!< Internal C entry point for creating IBuilder.

namespace nvinfer1
{
//!
//! \brief Create an instance of an IBuilder class.
//!
//! This class is the logging class for the builder.
//!
//! unnamed namespace avoids linkage surprises when linking objects built with different versions of this header.
//!
namespace
{
inline IBuilder* createInferBuilder(ILogger& logger)
{
    return static_cast<IBuilder*>(createInferBuilder_INTERNAL(&logger, NV_TENSORRT_VERSION));
}
}
}

#endif
