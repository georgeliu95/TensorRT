/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

/**
 * \brief TensorRT-Lite Public API Header.
 * This header file will be installed with the TensorRT-Lite library.
 */

#ifndef NVINFER_LITE_H
#define NVINFER_LITE_H

#include "NvInfer.h"

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvinfer1::lite
{

/**
 * \brief A tensor dimensions and data type tuple.
 */
using TensorInfo = std::pair<nvinfer1::Dims, nvinfer1::DataType>;
/**
 * \brief A hashmap for quering the tensor dimensions by the bound tensor name.
 * \details The key std::string is the name of the bindings.
 */
using DimsMap = std::unordered_map<std::string, nvinfer1::Dims>;
/**
 * \brief A hashmap for quering the tensor data type by the bound tensor name.
 */
using TypeMap = std::unordered_map<std::string, nvinfer1::DataType>;
/**
 * \brief A hashmap for quering the tensor dimensions and data type by the bound tensor name.
 */
using TensorMap = std::unordered_map<std::string, nvinfer1::lite::TensorInfo>;
/**
 * \brief A hashmap for quering the vector of input tensor profile shapes by the bound tensor name.
 */
using ProfileMap = std::unordered_map<std::string, std::vector<nvinfer1::Dims>>;
/**
 * \brief A hashmap for quering the pointer to the data buffer by the bound tensor name.
 */
using BufferMap = std::unordered_map<std::string, void*>;

/**
 * \brief TensorRT-Lite common deleter for nvinfer1::IExecutionContext and nvinfer1::ICudaEngine.
 * \details RAII for nvinfer1::IExecutionContext and nvinfer1::ICudaEngine.
 */
struct CommonDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

/**
 * \brief TensorRT tensor object.
 * It contains the pointer to tensor on memory and the memory host/device location.
 */
class NamedTensorPtr
{
public:
    /**
     * \brief Construct a new NamedTensorPtr object
     * \param ptrData Pointer to the memory that stores data.
     * \param location The location of the memory for data.
     * \param name Tensor name.
     */
    NamedTensorPtr(void* ptrData, const nvinfer1::TensorLocation location, const std::string& name)
        : mPtrData{ptrData}
        , mLocation{location}
        , mName{name}
    {
    }

    /**
     * \brief Construct a new NamedTensorPtr object
     * \param ptrData Pointer to the memory that stores data.
     * \param location The location of the memory for data.
     */
    NamedTensorPtr(void* ptrData, const nvinfer1::TensorLocation location) noexcept
        : NamedTensorPtr{ptrData, location, ""}
    {
    }

    /**
     * \brief Get the pointer to the tensor data.
     * \returns Pointer to the memory that stores data.
     */
    void* getDataPtr() const noexcept
    {
        return mPtrData;
    }

    /**
     * \brief Get the tensor location.
     * \returns The location of the memory for data.
     */
    nvinfer1::TensorLocation getTensorLocation() const noexcept
    {
        return mLocation;
    }

    /**
     * \brief Get the tensor name.
     * \returns Tensor name.
     */
    std::string getTensorName() const noexcept
    {
        return mName;
    }

private:
    void* mPtrData{nullptr};
    nvinfer1::TensorLocation mLocation{nvinfer1::TensorLocation::kDEVICE};
    std::string mName;
};

/**
 * \brief TensorRT-Lite logger.
 */
class Logger : public nvinfer1::ILogger
{
public:
    /**
     * \brief Construct a new Logger object.
     */
    Logger() = default;

    /**
     * \brief Construct a new Logger object.
     * \param enable Controls logging of messages.
     */
    explicit Logger(bool enable)
        : nvinfer1::ILogger{}
        , mEnable{enable}
    {
    }

    /**
     * \brief Log method override - print to standard error/output stream.
     * \param severity TensorRT severity. See
     * https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_logger.html
     * \param msg Log message.
     */
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        // The logger should always output kERROR and kWARNING messages.
        if (mEnable)
        {
            if (severity == nvinfer1::ILogger::Severity::kERROR)
            {
                std::cerr << msg << std::endl;
            }
            if (severity == nvinfer1::ILogger::Severity::kWARNING)
            {
                std::cout << msg << std::endl;
            }
        }
    }

    /**
     * \brief Enable logging.
     */
    void enable() noexcept
    {
        mEnable = true;
    }

    /**
     * \brief Disable logging.
     */
    void disable() noexcept
    {
        mEnable = false;
    }

private:
    bool mEnable{false};
};

/**
 * \brief TensorRT-Lite supported calibrator types.
 */
enum class CalibratorType : int32_t
{
    // TensorRT Entropy Calibrator V2 for determining the dynamic ranges of the tensor values.
    // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_int8_entropy_calibrator2.html
    kENTROPY_CALIBRATOR_V2 = 0,
    // TensorRT MinMax Calibrator for determining the dynamic ranges of the tensor values.
    // https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_int8_min_max_calibrator.html
    kMIN_MAX_CALIBRATOR = 1,
};

/**
 * \brief Base data stream to be derived and overridden for TensorRT-Lite INT8 calibration.
 */
class ICalibrationBaseStream
{
public:
    /**
     * \brief Check if the stream has ended or not.
     * This method has to be overridden.
     * \returns true if the stream still has data to load.
     * \returns false if the stream has ended.
     */
    virtual bool hasNext() noexcept = 0;

    /**
     * \brief Get the dimensions and the data type of the input from the stream.
     * This will be used for allocating the buffer on GPU.
     * \param bindingName The binding name.
     * \returns The input dimensions and data type.
     */
    virtual const nvinfer1::lite::TensorInfo& getInputInfo(const std::string& bindingName) const noexcept = 0;

    /**
     * \brief Get a batch of the data buffer on the host via pointers.
     * The data buffer on the host will be copied to the device by EngineBuilder.
     * This method might throw exceptions.
     * If the name string is longer than std::string::max_size(), throws std::length_error.
     * If the memory allocation fails, throws std::bad_alloc.
     * \returns The map from input binding name to input data.
     */
    virtual const std::unordered_map<std::string, const void*>& getBatch() = 0;

    /**
     * \brief Destroy the ICalibrationBaseStream object
     * Virtual destructor to allow the ICalibrationBaseStream object to be destroyed correctly with runtime
     * polymorphisms.
     */
    virtual ~ICalibrationBaseStream() = default;

protected:
    /**
     * \brief Construct a new ICalibrationBaseStream object
     * ICalibrationBaseStream has to be derived.
     */
    ICalibrationBaseStream() = default;
};

/**
 * \brief TensorRT-Lite engine builder optimization configurations.
 */
struct OptimizationConfig
{
    /**
     * \brief The file path to an ONNX model.
     * Only ONNX model is supported in TensorRT-Lite.
     */
    std::string onnxFilepath;

    /**
     * \brief The file path to plugin shared library files.
     */
    std::vector<std::string> pluginFilepaths;

    /**
     * \brief Specify whether to use DLA.
     * If useDLA is false then the value of dlaCore will be ignored.
     * If useDLA is true then the value of dlaCore has to be a valid value,
     * otherwise, runtime error will happen.
     */
    bool useDLA{false};

    /**
     * \brief Specify the DLA core to run network on.
     */
    int32_t dlaCore{-1};

    /**
     * \brief Allow runnning the network in Int8 mode.
     */
    bool int8{false};

    /**
     * \brief Allow running the network in FP16 mode.
     */
    bool fp16{true};

    /**
     * \brief Max workspace size for building engine.
     * Unit in MB.
     * Set default to 1 GB.
     */
    int32_t maxWorkspaceSize{1024};

    /**
     * \brief Multiple profiles for per engine supported.
     */
    std::vector<nvinfer1::lite::ProfileMap> profileShapes;

    /**
     * \brief Calibration profile.
     * This is required for engines using dynamic shapes.
     * No matter how many optimization profiles were provided, only exactly one calibration profile is required.
     * The shape of calibration profile could be different from the optimization profiles.
     */
    nvinfer1::lite::DimsMap calibrationProfileShapes;

    /**
     * \brief pointer to BaseStream inherited instances.
     * Users are responsible to override the BaseStream class
     */
    nvinfer1::lite::ICalibrationBaseStream* stream{nullptr};

    /**
     * \brief Filepath to read and load calibration cache
     */
    std::string calibrationCacheFilepath;

    /**
     * \brief Calibrator type for INT8 calibration.
     */
    nvinfer1::lite::CalibratorType calibratorType{};

    /**
     * \brief Use existing calibration cache if there is one.
     */
    bool useCalibrationCache{false};
};

/**
 * \brief TensorRT-Lite abstract engine builder.
 * Build a TensorRT engine using the optimization configuration.
 */
class IEngineBuilder
{
public:
    /**
     * \brief Set the engine building configurations.
     * \param configs An OptimizationConfig object.
     */
    virtual void setOptimizationConfig(const OptimizationConfig& configs) noexcept = 0;

    /**
     * \brief Build and return a pointer to TensorRT engine.
     * This method might throw exceptions.
     * If the binding name string is longer than std::string::max_size(), throws std::length_error.
     * If the memory allocation fails, throws std::bad_alloc.
     * If any of the building step fails, throws std::runtime_error with details.
     * \returns A unique pointer to TensorRT engine.
     */
    virtual std::unique_ptr<nvinfer1::ICudaEngine, nvinfer1::lite::CommonDeleter> buildEngine() = 0;

    /**
     * \brief Save TensorRT engine to hard drive.
     * This method might throw exceptions.
     * If any of the saving step fails, throws std::runtime_error with details.
     * \param engine A reference of the unique pointer to TensorRT engine.
     * \param engineFilepath A reference to the string of engine file path.
     */
    virtual void saveEngine(const std::unique_ptr<nvinfer1::ICudaEngine, nvinfer1::lite::CommonDeleter>& engine,
        const std::string& engineFilepath) const = 0;

    /**
     * \brief Load TensorRT engine from hard drive.
     * This method might throw exceptions.
     * If any of the saving step fails, throws std::runtime_error with details.
     * \param engineFilepath A reference to the string of engine file path.
     * \returns A unique pointer to TensorRT engine.
     */
    virtual std::unique_ptr<nvinfer1::ICudaEngine, nvinfer1::lite::CommonDeleter> loadEngine(
        const std::string& engineFilepath) const = 0;

    virtual ~IEngineBuilder() = default;

protected:
    IEngineBuilder() = default;
};

/**
 * \brief Create a unique pointer to the abstract engine builder.
 * \param logger A TensorRT logger.
 * \returns A unique pointer to the abstract engine builder.
 */
std::unique_ptr<nvinfer1::lite::IEngineBuilder> createEngineBuilder(std::unique_ptr<nvinfer1::ILogger>& logger) noexcept;

/**
 * \brief TensorRT-Lite abstract inference session.
 * Create a base inference session from a TensorRT engine.
 * Host and device buffer could be automatically managed.
 */
class IInferenceSession
{
public:
    /**
     * \brief Load TensorRT plugins.
     * This method is exposed to the user because the user could set the plugin file path in OptimizationConfig
     * objects. This method might throw exceptions. If the name string is longer than std::string::max_size(), throws
     * std::length_error. If the memory allocation fails, throws std::bad_alloc.
     * \param pluginFilepaths A vector of the plugin file path strings.
     */
    virtual void loadPlugins(const std::vector<std::string>& pluginFilepaths) = 0;

    /**
     * \brief Get the binding names for this TensorRT engine.
     * \returns The vector of binding names.
     */
    virtual const std::vector<std::string>& getBindingNames() const noexcept = 0;

    /**
     * \brief Get the input binding names for this TensorRT engine.
     * \returns The vector of input binding names.
     */
    virtual const std::vector<std::string>& getInputBindingNames() const noexcept = 0;

    /**
     * \brief Get the output binding names for this TensorRT engine.
     * \returns The vector of output binding names.
     */
    virtual const std::vector<std::string>& getOutputBindingNames() const noexcept = 0;

    /**
     * \brief Get the number of optimization profiles from the TensorRT engine.
     * \returns The number of optimization profiles from the TensorRT engine.
     */
    virtual int32_t getNbOptimizationProfiles() const noexcept = 0;

    /**
     * \brief Get the optimization profile of the input.
     * This method might throw exceptions.
     * If the optimizationProfileId does not exist, throws std::out_of_range.
     * \param optimizationProfileId The optimization profile ID.
     * \param bindingName The binding name.
     * \returns The length of the returned vector is [min shape, optimum shape, max shape].
     */
    virtual const std::vector<nvinfer1::lite::Dims>& getOptimizationProfileInputShape(
        const int32_t optimizationProfileId, const std::string& bindingName) const = 0;

    /**
     * \brief Set the optimization profile for the execution context.
     * By default, the constructor will call this method and set the optimization profile to 0.
     * This method might throw exceptions.
     * If the optimizationProfileId does not exist, throws std::out_of_range.
     * \param optimizationProfileId The optimization profile ID.
     * \returns true if the setting was successful.
     * \returns false if the setting was not successful.
     */
    virtual bool setActiveOptimizationProfile(const int32_t optimizationProfileId) = 0;

    /**
     * \brief Set all the input binding shape for the specific binding for the execution context.
     * This is required for inputs using dynamic shapes.
     * Input bindings that uses explicit dimension but without dynamic shapes could skip calling this method.
     * This method might throw exceptions.
     * If the binding name does not exist for the engine, throws std::runtime_error with details.
     * \param bindingName The binding name.
     * \param inputBindingShape The map from input names to input dimensions.
     * \returns true if the setting was successful.
     * \returns false if the setting was not successful.
     */
    virtual bool setInputBindingShape(const std::string& bindingName, const nvinfer1::lite::Dims& inputBindingShape) = 0;

    /**
     * \brief Get the binding shape via binding name.
     * All input shapes have to be set by default or by calling setInputBindingShape before calling this method.
     * This method might throw exceptions.
     * If the binding name does not exist for the engine, throws std::runtime_error with details.
     * \param bindingName The binding name.
     * \returns The dimension corresponding to the binding name.
     */
    virtual nvinfer1::Dims getBindingShape(const std::string& bindingName) const = 0;

    /**
     * \brief Get the binding data type via binding name.
     * This method might throw exceptions.
     * If the binding name does not exist for the engine, throws std::runtime_error with details.
     * \param bindingName The binding name.
     * \returns The data type corresponding to the binding name.
     */
    virtual nvinfer1::DataType getBindingDataType(const std::string& bindingName) const = 0;

    /**
     * \brief Get the binding info, including binding shape and data type, via binding name.
     * This method might throw exceptions.
     * If the binding name does not exist for the engine, throws std::runtime_error with details.
     * \param bindingName The binding name.
     * \returns The dimension and data type corresponding to the binding name.
     */
    virtual nvinfer1::lite::TensorInfo& getBindingInfo(const std::string& bindingName) const = 0;

    /**
     * \brief Allocate the largest buffer for all the bindings that would be used for inference.
     * The InferenceSession queries the TensorRT engine for the largest profile dimensions.
     * Allocate the buffer size corresponding to the largest profile dimensions.
     * This method is not called automatically in the constructors because sometimes the user would not use the largest
     * profile dimensions for inference. Thus allocating a lot of buffer is not useful.
     * This method might throw exceptions.
     * If the memory allocation fails, throws std::bad_alloc.
     */
    virtual void allocateBuffers() = 0;

    /**
     * \brief Allocate the buffer of user specified maximal sizes for the binding that would be used for inference.
     * If the runtime largest binding dimensions will be smaller than the largest profile dimensions from the TensorRT
     * engine, the user could set their own.
     * This method might throw exceptions.
     * If the memory allocation fails, throws std::bad_alloc.
     * If the binding name does not exist in all the binding names for the engine,
     * throw std::runtime_error with details.
     * \param bindingName The binding name.
     * \param maxBindingShapes The maximum binding shapes.
     */
    virtual void allocateBuffer(const std::string& bindingName, const nvinfer1::lite::Dims& maxBindingShape) = 0;

    /**
     * \brief Get the pointer to the host buffer via binding name.
     * This method might throw exceptions.
     * If the binding name does not exist for the engine, throws std::runtime_error with details.
     * \param bindingName The binding name.
     * \returns The pointer to the data corresponding to the binding name on host.
     */
    virtual void* getHostBufferPtr(const std::string& bindingName) const = 0;

    /**
     * \brief Get the pointer to the device buffer via binding name.
     * \param bindingName The binding name.
     * \returns The pointer to the data corresponding to the binding name on device.
     */
    virtual void* getDeviceBufferPtr(const std::string& bindingName) const noexcept = 0;

    /**
     * \brief Run TensorRT inference.
     * All the arguments are optional.
     * If none of the arguments were provided,
     * The inference session assumes all the input data has been copied to the host buffer that the inference session
     * owns, it would copy the data from the host buffer to the device buffer that the inference session owns, execute
     * inference, and copy the output data from the device buffer to the host buffer that the inference session owns.
     * The user could also provide custom buffers using NamedTensorPtr, which contains buffer location information, as
     * arguments. Depending on the NamedTensorPtr location, the inference session would automatically determine whether
     * copying memory from host to device and device to host are required. For example, if the buffer location of all
     * the inputTensors are on the GPU device, and the buffer location of all the outputTensors are also on the GPU
     * device, there will be no memory copy between host and device. If stream is provided, the inference will be
     * asynchronous (non-blocking). If stream is not provided, the inference will be synchronous (blocking).
     * \param inputTensors A map of the input tensors.
     * \param outputTensors A map of the output tensors.
     * \param stream CUDA stream created by the user.
     * \returns true if the inference was successful.
     * \returns false if the inference was not successful.
     */
    virtual bool runInference() noexcept = 0;

    virtual bool runInference(const std::unordered_map<std::string, nvinfer1::lite::NamedTensorPtr>& inputTensors,
        std::unordered_map<std::string, nvinfer1::lite::NamedTensorPtr>& outputTensors) noexcept = 0;

    virtual bool runInferenceAsync(cudaStream_t stream) noexcept = 0;

    virtual bool runInferenceAsync(const std::unordered_map<std::string, nvinfer1::lite::NamedTensorPtr>& inputTensors,
        std::unordered_map<std::string, nvinfer1::lite::NamedTensorPtr>& outputTensors, cudaStream_t stream) noexcept = 0;

    virtual ~IInferenceSession() = default;

protected:
    IInferenceSession() = default;
};

/**
 * \brief Create a unique pointer to the abstract inference session.
 * \param logger A TensorRT logger.
 * \param engine A TensorRT engine builder.
 * \returns A unique pointer to the abstract inference session.
 */
std::unique_ptr<nvinfer1::lite::IInferenceSession> createInferenceSession(std::unique_ptr<nvinfer1::ILogger>& logger,
    std::unique_ptr<nvinfer1::ICudaEngine, nvinfer1::lite::CommonDeleter>& engine) noexcept;

} // namespace nvinfer1::lite

#endif // NVINFER_LITE_H
