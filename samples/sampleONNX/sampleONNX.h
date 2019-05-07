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

#ifndef _SAMPLE_ONNX_H
#define _SAMPLE_ONNX_H

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <map>
#include <string>

#ifdef _WIN32
#include "..\common\windows\getopt.h"
#else
#include "internalEngineAPI.h"
#include "InternalAPI.h"
#include "getOptions.h"
#endif

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "logger.h"
#include "common.h"
#include "ppm_utils.h"

#include "sampleConfig.h"

using namespace std;

const std::string gSampleName = "TensorRT.sample_onnx";

namespace sampleONNX
{

void print_usage()
{
    std::cout << "Usage:" << std::endl;
    std::cout << "sample_onnx onnx_model.pb"
                 "\n"
                 "                [-i image_file]  (Currently only PPM and ASCII text file (default) with input data is supported)."
                 "\n"
                 "                [-f ASCII | PPM] (input file format, ASCII if not specified) "
                 "\n"
                 "                [-r reference file]  (A file with classification categories, required if PPM input)"
                 "\n"
                 "                [-o output file]  (A file with row output vector, required if the output format is ASCII)"
                 "\n"
                 "                [-k topK]  (Compare topK probabilities from the output file and the reference file)"
                 "\n"
                 "                [-e engine_file.trt]  (output TensorRT engine)"
                 "\n"
                 "                [-t onnx_model.pbtxt] (output ONNX text file without weights)"
                 "\n"
                 "                [-T onnx_model.pbtxt] (output ONNX text file with weights)"
                 "\n"
                 "                [-b max_batch_size]"
                 "\n"
                 "                [-w max_workspace_size_bytes]"
                 "\n"
                 "                [-d model_data_type_bit_depth] (32 => float32, 16 => float16, 8 => int8)"
                 "\n"
                 "                [-l] (list layers and their shapes)"
                 "\n"
                 "                [-c calibration list file (A file containing the list of image files used for int 8 calibraton )"
                 "\n"
                 "                [-n calibration_batch_size]"
                 "\n"
                 "                [-m max_number_of_calibration_batches]"
                 "\n"
                 "                [-s index_of_the_first_calibration_batch]"
                 "\n"
                 "                [-u DLA core index]"
                 "\n"
                 "                [-g] (debug mode)"
                 "\n"
                 "                [-v] (increase verbosity)"
                 "\n"
                 "                [-q] (decrease verbosity)"
                 "\n"
                 "                [-V] (show version information)"
                 "\n"
                 "                [-h] (show help)"
              << std::endl;
}

//...Converts int type into nvinfer1::DataType
nvinfer1::DataType convert2DataType(const int mModelDtypeNbits)
{
    switch (mModelDtypeNbits)
    {
    case 8:
        return nvinfer1::DataType::kINT8;
    case 16:
        return nvinfer1::DataType::kHALF;
    case 32:
        return nvinfer1::DataType::kFLOAT;
    default:
        std::string message = "Invalid model data type bit depth: " + std::to_string(mModelDtypeNbits);
        gLogError << message << std::endl;
        throw std::range_error(message);
        return nvinfer1::DataType::kINT8;
    } // end switch
}

template <typename APEX>
bool getopt(int argc, char* argv[], APEX& apex)
{
#ifdef TRT_DEBUG
    if (apex.isDebug())
    {
        gLogInfo << "APEX::getopt() ";
    }
#endif
// FIXME TRT-7126 Use portable getOptions() method for Windows as well
#ifdef _WIN32
    int arg;
    while ((arg = ::getopt(argc, argv, "e:b:w:t:T:d:D:i:r:o:f:k:c:n:m:s:u:lgvqVh")) != -1)
    {
        switch (arg)
        {
        case 'e':
            if (optarg)
            {
                apex.setEngineFileName(optarg);
                break;
            }
            else
            {
                gLogError << "-e flag requires argument" << std::endl;
                return false;
            }
        case 't':
            if (optarg)
            {
                apex.setTextFileName(optarg);
                break;
            }
            else
            {
                gLogError << "-t flag requires argument" << std::endl;
                return false;
            }
        case 'T':
            if (optarg)
            {
                apex.setFullTextFileName(optarg);
                break;
            }
            else
            {
                gLogError << "-T flag requires argument" << std::endl;
                return false;
            }
        case 'b':
            if (optarg)
            {
                apex.setMaxBatchSize(atoll(optarg));
                break;
            }
            else
            {
                gLogError << "-b flag requires argument" << std::endl;
                return false;
            }
        case 'w':
            if (optarg)
            {
                apex.setMaxWorkSpaceSize(atoll(optarg));
                break;
            }
            else
            {
                gLogError << "-w flag requires argument" << std::endl;
                return false;
            }
        case 'd':
            if (optarg)
            {
                apex.setModelDtype(convert2DataType(atoi(optarg)));
                break;
            }
            else
            {
                gLogError << "-d flag requires argument" << std::endl;
                return false;
            }
        case 'i':
            if (optarg)
            {
                apex.setImageFileName(optarg);
                break;
            }
            else
            {
                gLogError << "-i flag requires argument" << std::endl;
                return false;
            }
        case 'f':
            if (optarg)
            {
                string inputFormat(optarg);
                std::transform(inputFormat.begin(), inputFormat.end(), inputFormat.begin(), ::tolower);
                if (inputFormat.compare("ascii") == 0)
                    apex.setInputDataFormat(SampleConfig::InputDataFormat::kASCII);
                else if (inputFormat.compare("ppm") == 0)
                    apex.setInputDataFormat(SampleConfig::InputDataFormat::kPPM);
                else
                {
                    gLogError << "-f flag accepts either ASCII or PPM argument" << std::endl;
                    return false;
                }

                break;
            }
            else
            {
                gLogError << "-i flag requires argument" << std::endl;
                return false;
            }
        case 'r':
            if (optarg)
            {
                apex.setReferenceFileName(optarg);
                break;
            }
            else
            {
                gLogError << "-r flag requires argument" << std::endl;
                return false;
            }
        case 'o':
            if (optarg)
            {
                apex.setOutputFileName(optarg);
                break;
            }
            else
            {
                gLogError << "-o flag requires argument" << std::endl;
                return false;
            }
        case 'k':
            if (optarg)
            {
                apex.setTopK(std::stoul(optarg));
                break;
            }
            else
            {
                gLogError << "-k flag requires integer argument" << std::endl;
                return false;
            }
        case 'c':
            if (optarg)
            {
                apex.setCalibrationFileName(optarg);
                break;
            }
            else
            {
                gLogError << "-o flag requires argument" << std::endl;
                return false;
            }
        case 'n':
            if (optarg)
            {
                apex.setCalibBatchSize(std::stoul(optarg));
                break;
            }
            else
            {
                gLogError << "-n flag requires integer argument" << std::endl;
                return false;
            }
        case 'm':
            if (optarg)
            {
                apex.setMaxNCalibBatch(std::stoul(optarg));
                break;
            }
            else
            {
                gLogError << "-m flag requires integer argument" << std::endl;
                return false;
            }
        case 's':
            if (optarg)
            {
                apex.setFirstCalibBatch(std::stoul(optarg));
                break;
            }
            else
            {
                gLogError << "-n flag requires integer argument" << std::endl;
                return false;
            }
        case 'u':
            if (optarg)
            {
                apex.setUseDLACore(std::stoul(optarg));
                break;
            }
            else
            {
                gLogError << "-u flag requires integer argument" << std::endl;
                return false;
            }

        case 'l': apex.setPrintLayerInfo(true); break;
        case 'g':
            gLogWarning << "debug builder currently unsupported " << std::endl;
            break; //apex.setDebugBuilder(); break;
        case 'v': apex.addVerbosity(); break;
        case 'q': apex.reduceVerbosity(); break;
        case 'V': samplesCommon::print_version(); return true;
        case 'h': sampleONNX::print_usage(); exit(EXIT_FAILURE);
        }
    }
    int num_args = argc - optind;
    if (num_args != 1)
    {
        sampleONNX::print_usage();
        return false;
    }
    apex.setModelFileName(argv[optind]);
#else /* non-Windows build - use portable getOptions() instead of ::getopt() */
    std::map<char, int> kwOccurrences;
    std::map<char, const char*> kwValues;
    std::vector<const char*> posArgs;
    auto errCode = nvinfer1::utility::getOptions(argc, argv, "lgvqVh", "ebwtTdDirofkcnmsu", kwOccurrences, kwValues, posArgs);
    if (errCode != 0)
    {
        gLogError << "Invalid input argument " << argv[errCode] << std::endl;
        return false;
    }
    if (kwValues['e'])
    {
        apex.setEngineFileName(kwValues['e']);
    }
    if (kwValues['t'])
    {
        apex.setTextFileName(kwValues['t']);
    }
    if (kwValues['T'])
    {
        apex.setFullTextFileName(kwValues['T']);
    }
    if (kwValues['b'])
    {
        apex.setMaxBatchSize(atoll(kwValues['b']));
    }
    if (kwValues['w'])
    {
        apex.setMaxWorkSpaceSize(atoll(kwValues['w']));
    }
    if (kwValues['d'])
    {
        apex.setModelDtype(convert2DataType(atoi(kwValues['d'])));
    }
    if (kwValues['i'])
    {
        apex.setImageFileName(kwValues['i']);
    }
    if (kwValues['f'])
    {
        string inputFormat(kwValues['f']);
        std::transform(inputFormat.begin(), inputFormat.end(), inputFormat.begin(), ::tolower);
        if (inputFormat.compare("ascii") == 0)
        {
            apex.setInputDataFormat(SampleConfig::InputDataFormat::kASCII);
        }
        else if (inputFormat.compare("ppm") == 0)
        {
            apex.setInputDataFormat(SampleConfig::InputDataFormat::kPPM);
        }
        else
        {
            gLogError << "-f flag accepts either ASCII or PPM argument" << std::endl;
            return false;
        }
    }
    if (kwValues['r'])
    {
        apex.setReferenceFileName(kwValues['r']);
    }
    if (kwValues['o'])
    {
        apex.setOutputFileName(kwValues['o']);
    }
    if (kwValues['k'])
    {
        apex.setTopK(std::stoul(kwValues['k']));
    }
    if (kwValues['c'])
    {
        apex.setCalibrationFileName(kwValues['c']);
    }
    if (kwValues['n'])
    {
        apex.setCalibBatchSize(std::stoul(kwValues['n']));
    }
    if (kwValues['m'])
    {
        apex.setMaxNCalibBatch(std::stoul(kwValues['m']));
    }
    if (kwValues['s'])
    {
        apex.setFirstCalibBatch(std::stoul(kwValues['s']));
    }
    if (kwValues['u'])
    {
        apex.setUseDLACore(std::stoul(kwValues['u']));
    }
    if (kwOccurrences['l'] > 0)
    {
        apex.setPrintLayerInfo(true);
    }
    if (kwOccurrences['g'] > 0)
    {
        gLogWarning << "debug builder currently unsupported " << std::endl;
    }
    for (int i = 0, n = kwOccurrences['v']; i < n; ++i)
    {
        apex.addVerbosity();
    }
    for (int i = 0, n = kwOccurrences['q']; i < n; ++i)
    {
        apex.reduceVerbosity();
    }
    if (kwOccurrences['V'] > 0)
    {
        samplesCommon::print_version();
        return true;
    }
    if (kwOccurrences['h'] > 0)
    {
        sampleONNX::print_usage();
        exit(EXIT_FAILURE);
    }
    if (posArgs.size() != 1)
    {
        sampleONNX::print_usage();
        return false;
    }
    apex.setModelFileName(posArgs[0]);
#endif
#ifdef TRT_DEBUG
    if (apex.isDebug())
    {
        if (apex.getModelFileName())
        {
            gLogInfo << " the following options were received:" << std::endl;
            gLogInfo << "\t" << apex.getModelFileName() << std::endl;
        }
        else
        {
            gLogError << "empty ONNX file name" << std::endl;
        }
    }
#endif

    return true;
}

//...Check the options, after the getopt is called
template <typename APEX_t>
bool checkOpt(const APEX_t&)
{
    return true;
}

inline std::map<std::string, std::string> getInputOutputNames(const nvinfer1::ICudaEngine& trt_engine)
{
    int nbindings = trt_engine.getNbBindings();
    std::map<std::string, std::string> tmp;
    for (int b = 0; b < nbindings; ++b)
    {
        nvinfer1::Dims dims = trt_engine.getBindingDimensions(b);
        if (trt_engine.bindingIsInput(b))
        {
            gLogInfo << "Found input: "
                     << trt_engine.getBindingName(b)
                     << " shape=" << dims
                     << " dtype=" << (int) trt_engine.getBindingDataType(b)
                     << std::endl;
            tmp["input"] = trt_engine.getBindingName(b);
        }
        else
        {
            gLogInfo << "Found output: "
                     << trt_engine.getBindingName(b)
                     << " shape=" << dims
                     << " dtype=" << (int) trt_engine.getBindingDataType(b)
                     << std::endl;
            tmp["output"] = trt_engine.getBindingName(b);
        }
    }
    return tmp;
}

template <typename APEX_t>
vector<float> prepareInput(APEX_t& apex, nvinfer1::ICudaEngine& trt_engine)
{
    vector<float> ret;
    string input_file = apex.getImageFileName();
    std::map<string, string> InOut = sampleONNX::getInputOutputNames(trt_engine);
    const int inputIndex = trt_engine.getBindingIndex(InOut["input"].c_str());
    nvinfer1::Dims inp_dims = trt_engine.getBindingDimensions(inputIndex);
    assert(inp_dims.nbDims >= 3 && "Input must be a tensor of at least rank 3.");

    switch (apex.getInputDataFormat())
    {
    case SampleConfig::InputDataFormat::kPPM:
    {
        if (samplesCommon::toLower(samplesCommon::getFileType(input_file)).compare("ppm") != 0)
        {
            //...Throw an exception here.
            gLogError << "wrong fromat: " << input_file << " is not a ppm file. " << std::endl;
            return ret;
        }

        PPM::PPM ppm1(inp_dims.d[0], inp_dims.d[1], inp_dims.d[2]);
        PPM::readPPMFile(input_file, ppm1);
        PPM::Image<uint8_t> image(ppm1);

        image = resize_image(image, inp_dims.d[1], inp_dims.d[2]);
        image = hwc2chw(image);
        image = convert_to_float(image);
        float* tmp = reinterpret_cast<float*>(image.data.data());
        ret.reserve(image.getDataSize());
        ret.assign(tmp, tmp + image.getDataSize());
        gLogInfo << "returning vector of " << ret.size() << " size" << std::endl;

        samplesCommon::writeASCIIFile("./debugPPMOut.txt", ret);
    }

    break;
    case SampleConfig::InputDataFormat::kASCII:
    {
        // we assume it is in CHW
        if (!samplesCommon::readASCIIFile(input_file, inp_dims.d[2] * inp_dims.d[1] * inp_dims.d[0], ret))
        {
            gLogError << "Failed to read: " << input_file << " as an ASCII file " << std::endl;
            return ret;
        }
    }
    break;
    default:
        gLogError << "Unexpected Input Data Format" << std::endl;
    } // end switch
    return ret;
}

inline std::map<std::string, std::string> getInputOutputNames(const nvinfer1::ICudaEngine* trt_engine)
{
    return getInputOutputNames(*trt_engine);
}

inline int volume(Dims dims)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int>());
}

//template <typename image_t>
size_t inferenceImage(vector<float>& input,
                    nvinfer1::ICudaEngine& trt_engine,
                    void** buffers,
                    cudaStream_t& stream,
                    vector<float>& output,
                    const int batchSize)
{

    //...LG: there should be only an input and an output
    gLogInfo << "----- Preparing to run Inference " << std::endl;

    assert(trt_engine.getNbBindings() == 2);

    std::map<string, string> InOut = sampleONNX::getInputOutputNames(trt_engine);

    //for (auto item: InOut) { gLogInfo << item.first <<" :\t " << item.second << std::endl; }

    //... allocate memory for input/output tensor
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = trt_engine.getBindingIndex(InOut["input"].c_str()), outputIndex = trt_engine.getBindingIndex(InOut["output"].c_str());

    nvinfer1::Dims inp_dims = trt_engine.getBindingDimensions(inputIndex);
    nvinfer1::Dims output_dims = trt_engine.getBindingDimensions(outputIndex);

    size_t inputDim = volume(inp_dims);
    size_t outputDim = volume(output_dims);
    output.resize(outputDim*batchSize);

    // create GPU buffers and a stream
    size_t input_size = inputDim * sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * input_size));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * outputDim * sizeof(float)));

    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU and replicate it for all batches, execute the batch asynchronously, and DMA it back:
    for (int batch = 0; batch < batchSize; batch++)
    {
        CHECK(cudaMemcpyAsync(static_cast<char *>(buffers[inputIndex]) + batch*input_size, input.data(), input_size, cudaMemcpyHostToDevice, stream));
    }

    nvinfer1::IExecutionContext* context = trt_engine.createExecutionContext();
    gLogInfo << " ----- Inference is ready" << std::endl;
    gLogInfo << " ----- Running inference" << std::endl;
    //...Pre Timing
    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
    float ms;
    cudaEventRecord(start, stream);
    //...End of Timing

    context->enqueue(batchSize, buffers, stream, nullptr);

    //...Post Timing
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    //...End postTiming

    gLogInfo << "Run inference in " << ms << " ms." << std::endl;

    CHECK(cudaMemcpyAsync(output.data(), buffers[outputIndex], batchSize * outputDim * sizeof(float), cudaMemcpyDeviceToHost, stream));

    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaStreamDestroy(stream));

    CHECK(cudaFree(buffers[inputIndex]));

    CHECK(cudaFree(buffers[outputIndex]));

    context->destroy();

    return outputDim;
} //  void inferenceImage

inline void copyConfig(nvonnxparser::IOnnxConfig* src, nvonnxparser::IOnnxConfig* dst)
{
    if (strlen(src->getModelFileName()) != 0)
        dst->setModelFileName(src->getModelFileName());
    dst->setModelDtype(src->getModelDtype());
    dst->setVerbosityLevel(src->getVerbosityLevel());
    if (strlen(src->getTextFileName()) != 0)
        dst->setTextFileName(src->getTextFileName());
    if (strlen(src->getFullTextFileName()) != 0)
        dst->setFullTextFileName(src->getFullTextFileName());
    dst->setPrintLayerInfo(src->getPrintLayerInfo());
}

} // end  namespace sampleONNX

#endif
