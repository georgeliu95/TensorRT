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

#ifndef SAMPLE_PARSE_ARGS_H
#define SAMPLE_PARSE_ARGS_H

#include "NvInfer.h"
#include "common.h"

#ifdef _MSC_VER
#include "..\common\windows\getopt.h"
#else
#include <getopt.h>
#endif
#include <iostream>
#include <string>
#include <vector>

enum class NetworkFormat : int
{
    kCAFFE = 0,
    kONNX = 1,
    kUFF = 2,
    kTRT = 3,
};

struct SampleArgs
{
    bool help{false};
    std::vector<std::string> dataDirs;
    NetworkFormat networkFormat{NetworkFormat::kCAFFE};
    std::string caffePrototxtFileName{"ResNet50_N2.prototxt"}, caffeWeightsFileName{"ResNet50_fp32.caffemodel"};
    std::string onnxFileName;
    std::string uffFileName;
    std::string trtFileName;
    std::string outputPath{"serializedNetwork.tnb"};
    std::vector<std::string> inputTensorNames, outputTensorNames;
    std::vector<std::vector<int>> inputTensorShapes;
    bool runInference{false};
};

bool parseArgs(int argc, char** argv, SampleArgs& args)
{
    const struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"datadir", required_argument, 0, 'd'},
        {"format", required_argument, 0, 'f'},
        {"caffe_prototxt", required_argument, 0, 'p'},
        {"caffe_weights", required_argument, 0, 'w'},
        {"onnx_file", required_argument, 0, 'n'},
        {"uff_file", required_argument, 0, 'u'},
        {"trt_file", required_argument, 0, 't'},
        {"output", required_argument, 0, 'o'},
        {"input_name", required_argument, 0, 'i'},
        {"output_name", required_argument, 0, 'j'},
        {"input_shape", required_argument, 0, 's'},
        {"run_inference", no_argument, 0, 'r'},
        {nullptr, 0, nullptr, 0}};

    while (1)
    {
        int arg;
        int option_index = 0;
        arg = getopt_long(argc, argv, "hd:f:p:w:n:u:t:o:i:j:s:r", long_options, &option_index);
        if (arg == -1)
        {
            break;
        }

        switch (arg)
        {
        case 'h':
            args.help = true;
            return false;
        case 'd':
            if (optarg)
            {
                args.dataDirs.push_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }
            break;
        case 'f':
            if (optarg)
            {
                std::string optargStr = optarg;
                if (optargStr == "caffe")
                {
                    args.networkFormat = NetworkFormat::kCAFFE;
                }
                else if (optargStr == "onnx")
                {
                    args.networkFormat = NetworkFormat::kONNX;
                }
                else if (optargStr == "uff")
                {
                    args.networkFormat = NetworkFormat::kUFF;
                }
                else if (optargStr == "trt")
                {
                    args.networkFormat = NetworkFormat::kTRT;
                }
                else
                {
                    std::cerr << "ERROR: --format argument must be one of: caffe, onnx, uff" << std::endl;
                    return false;
                }
            }
            else
            {
                std::cerr << "ERROR: --format requires option argument" << std::endl;
                return false;
            }
            break;
        case 'p':
            if (optarg)
            {
                args.caffePrototxtFileName = optarg;
            }
            else
            {
                std::cerr << "ERROR: --caffe_prototxt requires option argument" << std::endl;
                return false;
            }
            break;
        case 'w':
            if (optarg)
            {
                args.caffeWeightsFileName = optarg;
            }
            else
            {
                std::cerr << "ERROR: --caffe_weights requires option argument" << std::endl;
                return false;
            }
            break;
        case 'n':
            if (optarg)
            {
                args.onnxFileName = optarg;
            }
            else
            {
                std::cerr << "ERROR: --onnx_file requires option argument" << std::endl;
                return false;
            }
            break;
        case 'u':
            if (optarg)
            {
                args.uffFileName = optarg;
            }
            else
            {
                std::cerr << "ERROR: --uff_file requires option argument" << std::endl;
                return false;
            }
            break;
        case 't':
            if (optarg)
            {
                args.trtFileName = optarg;
            }
            else
            {
                std::cerr << "ERROR: --trt_file requires option argument" << std::endl;
                return false;
            }
            break;
        case 'o':
            if (optarg)
            {
                args.outputPath = optarg;
            }
            else
            {
                std::cerr << "ERROR: --output requires option argument" << std::endl;
                return false;
            }
            break;
        case 'i':
            if (optarg)
            {
                args.inputTensorNames.push_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --input_name requires option argument" << std::endl;
                return false;
            }
            break;
        case 'j':
            if (optarg)
            {
                args.outputTensorNames.push_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --output_name requires option argument" << std::endl;
                return false;
            }
            break;
        case 's':
            if (optarg)
            {
                std::vector<std::string> shapeStr = samplesCommon::splitString(optarg);
                std::vector<int> shape(shapeStr.size());
                std::transform(shapeStr.begin(), shapeStr.end(), shape.begin(), [](std::string s) { return atoi(s.c_str()); });
                args.inputTensorShapes.push_back(shape);
            }
            else
            {
                std::cerr << "ERROR: --input_shape requires option argument" << std::endl;
                return false;
            }
            break;
        case 'r':
            args.runInference = true;
            break;
        default:
            return false;
        }
    }

    if (args.networkFormat == NetworkFormat::kCAFFE && (args.caffePrototxtFileName == "" || args.caffeWeightsFileName == ""))
    {
        std::cerr << "ERROR: if you are using caffe, you have to provide --caffe_prototxt and --caffe_weights" << std::endl;
        return false;
    }
    if (args.networkFormat == NetworkFormat::kONNX && args.onnxFileName == "")
    {
        std::cerr << "ERROR: if you are using ONNX, you have to provide --onnx_file" << std::endl;
        return false;
    }
    if (args.networkFormat == NetworkFormat::kUFF)
    {
        if (args.uffFileName == "")
        {
            std::cerr << "ERROR: if you are using UFF, you have to provide --uff_file" << std::endl;
            return false;
        }
        if (args.inputTensorNames.size() != args.inputTensorShapes.size())
        {
            std::cerr << "ERROR: number of input_names and input_tensors doesn\'t match" << std::endl;
            return false;
        }
    }

    return true;
}

#endif // SAMPLE_PARSE_ARGS_H
