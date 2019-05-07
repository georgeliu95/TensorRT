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

#include "NvOnnxParser.h"
#include <iostream>

//...this file is from the old onnx, move it elsewhere (temprorary to sample directory)
#include "parserOnnxConfig.h"

#include "sampleONNX.h"

#include "common.h"
#include "ppm_utils.h"

#include "sampleConfig.h"

//...Int8 calibration
#include "BatchStream.h"
#include "Int8EntropyCalibrator.h"

#include <ctime>

using namespace std;
static int gUseDLACore{-1};

//... Move to common area.
void print_usage()
{
    cout << "ONNX to TensorRT model parser" << endl;
    cout << "Usage: sampleONNX onnx_model.pb"
         << "\n"
         << "                [-p input_file.ppm]  (input file in PPM format)"
         << "\n"
         << "                [-o engine_file.trt]  (output TensorRT engine)"
         << "\n"
         << "                [-t onnx_model.pbtxt] (output ONNX text file without weights)"
         << "\n"
         << "                [-T onnx_model.pbtxt] (output ONNX text file with weights)"
         << "\n"
         << "                [-b max_batch_size (default 32)]"
         << "\n"
         << "                [-w max_workspace_size_bytes (default 1 GiB)]"
         << "\n"
         << "                [-d model_data_type_bit_depth] (32 => float32, 16 => float16)"
         << "\n"
         << "                [-l] (list layers and their shapes)"
         << "\n"
         << "                [-g] (debug mode)"
         << "\n"
         << "                [-v] (increase verbosity)"
         << "\n"
         << "                [-q] (decrease verbosity)"
         << "\n"
         << "                [-V] (show version information)"
         << "\n"
         << "                [-h] (show help)" << endl;
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs{"data/samples/ssd/", "data/samples/ssd/VOC2007/",
                                  "data/samples/ssd/VOC2007/PPMImages/"};
    return locateFile(input, dirs);
}

int main(int argc, char** argv)
{
    auto sampleTest = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(sampleTest);

    // set logger reportable severity to kWARNING initially
    setReportableSeverity(Severity::kWARNING);

    SampleConfig* apex = new SampleConfig;
    BatchStream* calibrationStream = nullptr;
    Int8EntropyCalibrator2* calibrator = nullptr;

    bool status = sampleONNX::getopt(argc, argv, *apex);
    if (!status)
    {
        gLogError << "Failed to pass arguments " << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    if (!sampleONNX::checkOpt(*apex))
    {
        gLogError << "Passed arguments are inconsistent, exiting ..." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    setReportableSeverity(static_cast<Logger::Severity>(static_cast<int>(apex->getVerbosityLevel())));

    gLogInfo << "Parsed command line arguments " << std::endl;

    std::string engine_filename = apex->getEngineFileName();
    std::string text_filename = apex->getTextFileName();
    std::string full_text_filename = apex->getFullTextFileName();
    int max_batch_size = apex->getMaxBatchSize();

    size_t max_workspace_size = 1 << 30;
    bool print_layer_info = false;
    bool debug_builder = false;

    //...Suppress Compiler warning
    (void) debug_builder;

    std::string onnx_filename = apex->getModelFileName();
    const nvinfer1::DataType dataType = apex->getModelDtype();

    gLogInfo << "Parsing onnx file " << onnx_filename << std::endl;

    gLogInfo << " ----- Preparing the Builder ---- " << endl;
    auto trt_builder = samplesCommon::infer_object(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    assert(trt_builder != nullptr);
    auto trt_config = samplesCommon::infer_object(trt_builder->createNetworkConfig());
    assert(trt_config != nullptr);

    trt_builder->setMaxBatchSize(max_batch_size);
    trt_config->setMaxWorkspaceSize(apex->getMaxWorkSpaceSize());

    if (apex->getModelDtype() == nvinfer1::DataType::kHALF)
    {
        if (!trt_builder->platformHasFastFp16())
        {
            //...No FP 16 support
            gLogWarning << "This platform does not support Fast FP16" << std::endl;
            return gLogger.reportWaive(sampleTest);
        }

        gLogInfo << "Building TensorRT engine, FP16 available:" << endl;
        gLogInfo << "    Max batch size:     " << max_batch_size << endl;
        gLogInfo << "    Max workspace size: " << max_workspace_size / (1024. * 1024) << " MiB" << std::endl;

        trt_config->setFlag(BuilderFlag::kFP16);
    }
    else if (apex->getModelDtype() == nvinfer1::DataType::kINT8)
    { // INT 8

        int CAL_BATCH_SIZE = apex->getCalibBatchSize();
        int NB_CAL_BATCHES = apex->getMaxNCalibBatch();
        int FIRST_CAL_BATCH = apex->getFirstCalibBatch();

        string calibFilesList = apex->getCalibrationFileName();

        if (!calibFilesList.size())
        {
            gLogError << "A file containing the list of calibration images must be specified for int8 inference" << std::endl;
            return gLogger.reportFail(sampleTest);
        }

        BatchStream* calibrationStream = new BatchStream(CAL_BATCH_SIZE,
                                                         NB_CAL_BATCHES,
                                                         calibFilesList);
        Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(*calibrationStream, FIRST_CAL_BATCH);

        gLogInfo << "&calibrator = " << calibrator << endl;

        if (dataType == nvinfer1::DataType::kINT8)
        {
            trt_config->setFlag(BuilderFlag::kINT8);
        }

        trt_config->setInt8Calibrator(calibrator);

        if (!trt_builder->platformHasFastInt8())
        {
            gLogWarning << "Platform does not support Int8 " << std::endl;
            return gLogger.reportWaive(sampleTest);
        }
    } // END of int8

    //trt_builder->setDebugSync(debug_builder);

    gLogInfo << " ----- Builder is Done ---- " << std::endl;
    auto trt_network = samplesCommon::infer_object(trt_builder->createNetwork());
    gLogInfo << " ----- Created an Empty Network  ---- " << std::endl;
    auto trt_parser = samplesCommon::infer_object(nvonnxparser::createParser(*trt_network, gLogger.getTRTLogger()));
    gLogInfo << " ----- Created an ONNX Parser  ---- " << std::endl;

    // if( print_layer_info ) {
    //   trt_parser->setLayerInfoStream(&std::cout);
    // }

    //...This is just to supprese a warning about unused
    (void) print_layer_info;

    gLogInfo << "Parsing model" << endl;

    if (!trt_parser->parseFromFile(onnx_filename.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
    {
        gLogError << "Failed to parse onnx model from " << onnx_filename << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    gLogInfo << " ----- Creating Engine ---- " << std::endl;
    gUseDLACore = apex->getUseDLACore();
    samplesCommon::enableDLA(trt_builder.get(), trt_config.get(), gUseDLACore);

    auto t_start = std::chrono::high_resolution_clock::now(); //...Timer starts ...

    auto trt_engine = samplesCommon::infer_object(trt_builder->buildEngineWithConfig(*trt_network.get(), *trt_config.get()));

    auto t_end = std::chrono::high_resolution_clock::now(); //...Timer ends ...
    float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    gLogInfo << " ----- Engine is built in " << ms << " ms. ---- " << std::endl;

    if (!engine_filename.empty())
    {
        gLogInfo << "Writing TensorRT engine to " << engine_filename << std::endl;
        auto engine_plan = samplesCommon::infer_object(trt_engine->serialize());
        std::ofstream engine_file(engine_filename.c_str(), std::ios::binary);
        engine_file.write((char*) engine_plan->data(), engine_plan->size());
        engine_file.close();
    }

    //...Inference

    void* buffers[2];
    cudaStream_t stream;
    size_t outputDim;
    vector<float> output;
    {
        vector<float> input = sampleONNX::prepareInput(*apex, *trt_engine);
        if (!input.size())
        {
            gLogError << "The input tensor is of zero size - please check your path to the input or the file type" << std::endl;
            return gLogger.reportFail(sampleTest);
        }
        outputDim = sampleONNX::inferenceImage(input, *trt_engine, buffers, stream, output, max_batch_size);

    } // End of Inference

    gLogInfo << "All done" << std::endl;

    const size_t topK = apex->getTopK();
    bool failureSeen{false};

    // Dump layer information if -l option is set, Linux only.
#ifndef _WIN32
    if (apex->getPrintLayerInfo())
    {
        dumpLayerInformation(trt_engine.get(), gLogInfo);
    }
#endif

    switch (apex->getInputDataFormat())
    {
    case SampleConfig::InputDataFormat::kPPM:
    {
        vector<string> referenceVector;
        if (!samplesCommon::readReferenceFile(apex->getReferenceFileName(), referenceVector))
        {
            return gLogger.reportFail(sampleTest);
        }

        // Calculate topK for each batch independently from the data for first batch.
        // Note that the inputs, and therefore the outputs are identical for all batches.
        for (int batch = 0; batch < max_batch_size; batch++)
        {
            size_t offset = batch*outputDim;
            vector<float> outputSlice;
            outputSlice.assign(output.begin() + offset, output.begin() + offset + outputDim);

            auto inds = samplesCommon::argsort(output.cbegin() + offset, output.cbegin() + offset + outputDim - 1, true);

            for (size_t i = 0; i < topK; ++i)
            {
                inds[i] += offset;
                cout << inds[i] + 1 << ": " << output[inds[i]] << endl;
            }

            vector<string> top1Result = samplesCommon::classify(referenceVector, outputSlice, 1);
            for (auto result : top1Result)
            {
                cout << "SampleONNX result: Detected: " << result << endl;
            }
        }
    }
    break;
    case SampleConfig::InputDataFormat::kASCII:
    {
        string input_file = apex->getImageFileName();
        string outFile = apex->getOutputFileName();
        // Tolerance checking for inference
        float tolerance = (apex->getModelDtype() == nvinfer1::DataType::kFLOAT) ? 0.03f : 0.05f;
        if ((apex->getReferenceFileName() != nullptr) && (topK != 0))
        {
            vector<float> goldReference;
            // Golden reference is only specified for the first batch.
            if (!samplesCommon::readASCIIFile(apex->getReferenceFileName(), outputDim, goldReference))
            {
                gLogError << "Failed to read reference file " << string(apex->getReferenceFileName()) << std::endl;
                return gLogger.reportFail(sampleTest);
            }

            vector<size_t> refere_V = samplesCommon::topK(goldReference, topK);
            vector<size_t> result_V;
            result_V.reserve(topK * max_batch_size);
            const bool printResults{apex->getVerbosityLevel() >= static_cast<nvonnxparser::IOnnxConfig::Verbosity>(nvinfer1::ILogger::Severity::kINFO)};

            // Calculate topK for each batch independently from the data for first batch.
            // Note that the inputs, and therefore the outputs are identical for all batches.
            for (int batch = 0; batch < max_batch_size; batch++)
            {
                size_t offset = batch * outputDim;

                vector<float> outputSlice;
                outputSlice.assign(output.begin() + offset, output.begin() + offset + outputDim);
                vector<size_t> resultSlice = samplesCommon::topK(outputSlice, topK);

                for (size_t i = 0; i < topK; ++i)
                {
                    const auto expected = refere_V[i];
                    const auto actual = result_V[i + topK * batch] = resultSlice[i];

                    const auto diff = abs(outputSlice[actual] - outputSlice[expected]) / outputSlice[expected];

                    if (diff > tolerance)
                    {
                        gLogError << "Found mismatch: Top " << i + 1 << ": " << actual + offset << "\t" << expected + offset << std::endl;
                        gLogError << "Diff: " << diff << "\t" << "Tolerance: " << tolerance << std::endl;
                        gLogInfo << outputSlice[actual] << "\t" << outputSlice[expected] << std::endl;
                        failureSeen = true;
                    }
                    else if (printResults)
                    {
                        gLogInfo << "Top " << i + 1 << ": Inference index: " << actual + offset << "\tReference index:" << expected + offset << std::endl;
                        gLogInfo << "Inference value: "<< outputSlice[actual] << "\tReference value: " << outputSlice[expected] << std::endl;
                    }
                }
            }

            string topKFile = "topK.txt";
            if (!samplesCommon::writeASCIIFile(topKFile, result_V))
            {
                gLogError << "Failed to write topK file " << topKFile << std::endl;
                return gLogger.reportFail(sampleTest);
            }
        }

        if (!samplesCommon::writeASCIIFile(outFile, output))
        {
            gLogError << "Failed to write the output file " << outFile << std::endl;
            return gLogger.reportFail(sampleTest);
        }
        else
        {
            gLogInfo << "Wrote output file  " << outFile << std::endl;
        }

        break;
    }
    default:
    {
        gLogError << "Unexpected Input Data Format" << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    } // end switch

    if (apex->getModelDtype() == nvinfer1::DataType::kINT8)
    {
        delete calibrator;
        delete calibrationStream;
    }

    apex->destroy();

    gLogInfo << "Graceful exit ..." << std::endl;

    if (failureSeen) {
        return gLogger.reportFail(sampleTest);
    } else {
        return gLogger.reportPass(sampleTest);
    }
}
