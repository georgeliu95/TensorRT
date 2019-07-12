/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "NvInfer.h"

#include "sampleOptions.h"

namespace sample
{

Arguments argsToArgumentsMap(int argc, char* argv[])
{
    Arguments arguments;
    for (int i = 1; i < argc; ++i)
    {
        auto valuePtr = strchr(argv[i], '=');
        if (valuePtr)
        {
            std::string value{valuePtr+1};
            arguments.emplace(std::string(argv[i], valuePtr-argv[i]), value);
        }
        else
        {
            arguments.emplace(argv[i], "");
        }
    }
    return arguments;
}

void BaseModelOptions::parse(Arguments& arguments)
{
    if (checkEraseOption(arguments, "--onnx", model))
    {
        format = ModelFormat::kONNX;
    }
    else if (checkEraseOption(arguments, "--uff", model))
    {
        format = ModelFormat::kUFF;
    }
    else if (checkEraseOption(arguments, "--model", model))
    {
        format = ModelFormat::kCAFFE;
    }
}

void UffInput::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--uffNHWC", NHWC);
    std::vector<std::string> args;
    if (checkEraseRepeatedOption(arguments, "--uffInput", args))
    {
        for (const auto& i: args)
        {
            std::vector<std::string> values{multiOptionToStrings(i)};
            if (values.size() == 4)
            {
                nvinfer1::Dims3 dims{std::stoi(values[1]), std::stoi(values[2]), std::stoi(values[3])};
                inputs.emplace_back(values[0], dims);
            }
            else
            {
                throw std::invalid_argument(std::string("Invalid uffInput ") + i);
            }
        }
    }
}

void ModelOptions::parse(Arguments& arguments)
{
    baseModel.parse(arguments);

    switch (baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        checkEraseOption(arguments, "--deploy", prototxt);
        break;
    }
    case ModelFormat::kUFF:
    {
        uffInputs.parse(arguments);
        break;
    }
    case ModelFormat::kONNX:
        break;
    case ModelFormat::kANY:
    {
        if (checkEraseOption(arguments, "--deploy", prototxt))
        {
            baseModel.format = ModelFormat::kCAFFE;
        }
        break;
    }
    }

    if (baseModel.format == ModelFormat::kCAFFE || baseModel.format == ModelFormat::kUFF)
    {
        std::vector<std::string> outArgs;
        if (checkEraseRepeatedOption(arguments, "--output", outArgs))
        {
            for (const auto& o: outArgs)
            {
                for (auto& v: multiOptionToStrings(o))
                {
                    outputs.emplace_back(std::move(v));
                }
            }
        }
    }
}

void BuildOptions::parse(Arguments& arguments)
{
    int min{0};
    int opt{0};
    int max{0};
    checkEraseOption(arguments, "--minBatch", min);
    checkEraseOption(arguments, "--maxBatch", max);
    checkEraseOption(arguments, "--optBatch", opt);
    if (min)
    {
        minBatch = min;
    }
    if (max)
    {
        maxBatch = max;
    }
    else
    {
        maxBatch = minBatch;
    }
    if (opt)
    {
        optBatch = opt;
    }
    else
    {
        optBatch = maxBatch;
    }

    auto getFormats = [&arguments](std::vector<IOFormat>& formatsVector, const char* argument)
    {
        std::string list;
        checkEraseOption(arguments, argument, list);
        std::vector<std::string> formats{multiOptionToStrings(list)};
        for (const auto& f : formats)
        {
            formatsVector.push_back(stringToValue<IOFormat>(f));
        }

    };

    getFormats(inputFormats, "--inputIOFormats");
    getFormats(outputFormats, "--outputIOFormats");

    checkEraseOption(arguments, "--workspace", workspace);
    checkEraseOption(arguments, "--minTiming", minTiming);
    checkEraseOption(arguments, "--avgTiming", avgTiming);
    checkEraseOption(arguments, "--fp16", fp16);
    checkEraseOption(arguments, "--int8", int8);
    checkEraseOption(arguments, "--safe", safe);
    checkEraseOption(arguments, "--calib", calibration);
    if (checkEraseOption(arguments, "--loadEngine", engine))
    {
        load = true;
    }
    if (checkEraseOption(arguments, "--saveEngine", engine))
    {
        save = true;
    }
}

void SystemOptions::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--device", device);
    checkEraseOption(arguments, "--useDLACore", DLACore);
    checkEraseOption(arguments, "--allowGPUFallback", fallback);
    std::string pluginName;
    while (checkEraseOption(arguments, "--plugins", pluginName))
    {
        plugins.emplace_back(pluginName);
    }
}

void InferenceOptions::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--batch", batch);
    checkEraseOption(arguments, "--streams", streams);
    checkEraseOption(arguments, "--iterations", iterations);
    checkEraseOption(arguments, "--duration", duration);
    checkEraseOption(arguments, "--warmUp", warmup);
    checkEraseOption(arguments, "--threads", threads);
    checkEraseOption(arguments, "--useCudaGraph", graph);
    checkEraseOption(arguments, "--buildOnly", skip);
}

void ReportingOptions::parse(Arguments& arguments)
{
    checkEraseOption(arguments, "--percentile", percentile);
    checkEraseOption(arguments, "--avgRuns", avgs);
    checkEraseOption(arguments, "--verbose", verbose);
    checkEraseOption(arguments, "--dumpOutput", output);
    checkEraseOption(arguments, "--dumpProfile", profile);
    checkEraseOption(arguments, "--exportTimes", exportTimes);
    checkEraseOption(arguments, "--exportProfile", exportProfile);
}

bool parseHelp(Arguments& arguments)
{
    bool help{false};
    checkEraseOption(arguments, "--help", help);
    return help;
}

void AllOptions::parse(Arguments& arguments)
{
    model.parse(arguments);
    build.parse(arguments);
    system.parse(arguments);
    inference.parse(arguments);
    if (build.minBatch == defaultMinBatch && build.minBatch == build.maxBatch &&
        build.optBatch == build.maxBatch && inference.batch > defaultMinBatch)
    {
        build.minBatch = inference.batch;
        build.optBatch = inference.batch;
        build.maxBatch = inference.batch;
    }
    reporting.parse(arguments);
    helps = parseHelp(arguments);
}

bool ModelOptions::isValid(std::ostream& err) const
{
    switch (baseModel.format)
    {
    case ModelFormat::kUFF:
    {
        if (uffInputs.inputs.empty())
        {
            err << "Uff models require at least one input" << std::endl;
            return false;
        }
    }
    case ModelFormat::kCAFFE:
    {
        if (outputs.empty())
        {
            err << "Caffe and Uff models require at least one output" << std::endl;
            return false;
        }
    }
    case ModelFormat::kONNX:
        break;
    case ModelFormat::kANY:
    {
        err << "Model format not recognized, only Caffe, Uff, and ONNX models are supported" << std::endl;
        return false;
    }
    }

    return true;
}

bool BuildOptions::isValid(std::ostream& err) const
{
    if (load && save)
    {
        err << "saveEngine and loadEngine arguments cannot be specified at the same time" << std::endl;
        return false;
    }
    if (!(minBatch > 0 && minBatch <= optBatch && optBatch <= maxBatch))
    {
        err << "Inconsistent min, opt, and max batch size specification" << std::endl;
        return false;
    }
    return true;
}

bool ReportingOptions::isValid(std::ostream& err) const
{
    if (percentile < 0 || percentile > 100)
    {
        err << "percentile requested is negative or greater than 100" << std::endl;
        return false;
    }
    return true;
}

bool AllOptions::isValid(std::ostream& err) const
{
    if (!helps)
    {
        if (!build.isValid(err))
        {
            return false;
        }
        if (!build.load && !model.isValid(err))
        {
            err << "No valid model specification (caffe, uff, or onnx) or engine is provided" << std::endl;
            return false;
        }
        if (!build.load && (inference.batch < build.minBatch || inference.batch > build.maxBatch))
        {
            err << "Inference batch size is outside the range specified to build the engine" << std::endl;
            return false;
        }


        return reporting.isValid(err);
    }

    return true;
}

std::ostream& operator<<(std::ostream& os, const BaseModelOptions& options)
{
    os << "=== Model Options ===" << std::endl;

    os << "Format: ";
    switch (options.format)
    {
    case ModelFormat::kCAFFE:
    {
        os << "Caffe";
        break;
    }
    case ModelFormat::kONNX:
    {
        os << "ONNX";
        break;
    }
    case ModelFormat::kUFF:
    {
        os << "UFF";
        break;
    }
    case ModelFormat::kANY:
        os << "*";
        break;
    }
    os << std::endl << "Model: " << options.model << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const UffInput& input)
{
    os << "Uff Inputs Layout: " << (input.NHWC ? "NHWC" : "NCHW") << std::endl;
    for (const auto& i : input.inputs)
    {
        os << "Input: " << i.first << "," << i.second.d[0] << "," << i.second.d[1] << "," << i.second.d[2] << std::endl;
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const ModelOptions& options)
{
    os << options.baseModel;
    switch (options.baseModel.format)
    {
    case ModelFormat::kCAFFE:
    {
        os << "Prototxt: " << options.prototxt;
        break;
    }
    case ModelFormat::kUFF:
    {
        os << options.uffInputs;
        break;
    }
    case ModelFormat::kONNX: // Fallthrough: No options to report for ONNX or the generic case
    case ModelFormat::kANY:
        break;
    }

    os << "Output:";
    for (const auto& o : options.outputs)
    {
        os << " " << o;
    }
    os << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const IOFormat& format)
{
    switch(format.first)
    {
    case nvinfer1::DataType::kFLOAT:
    {
        os << "fp32:";
        break;
    }
    case nvinfer1::DataType::kHALF:
    {
        os << "fp16:";
        break;
    }
    case nvinfer1::DataType::kINT8:
    {
        os << "int8:";
        break;
    }
    case nvinfer1::DataType::kINT32:
    {
        os << "int32:";
        break;
    }
    }

    for(int f = 0; f < nvinfer1::EnumMax<nvinfer1::TensorFormat>(); ++f)
    {
        if ((1U<<f) & format.second)
        {
            if (f)
            {
                os << "+";
            }
            switch(nvinfer1::TensorFormat(f))
            {
            case nvinfer1::TensorFormat::kLINEAR:
            {
                os << "chw";
                break;
            }
            case nvinfer1::TensorFormat::kCHW2:
            {
                os << "chw2";
                break;
            }
            case nvinfer1::TensorFormat::kHWC8:
            {
                os << "hwc8";
                break;
            }
            case nvinfer1::TensorFormat::kCHW4:
            {
                os << "chw4";
                break;
            }
            case nvinfer1::TensorFormat::kCHW16:
            {
                os << "chw16";
                break;
            }
            case nvinfer1::TensorFormat::kCHW32:
            {
                os << "chw32";
                break;
            }
            }
        }
    }
    return os;
};

std::ostream& operator<<(std::ostream& os, const BuildOptions& options)
{
    auto printIOFormats = [](std::ostream& os, const char* direction, const std::vector<IOFormat> formats)
    {
        if (formats.empty())
        {
            os << direction << "s format: fp32:CHW" << std::endl;
        }
        else
        {
            for(const auto& f : formats)
            {
                os << direction << ": " << f << std::endl;
            }
        }
    };

// clang-format off
    os << "=== Build Options ==="                                                                                       << std::endl <<

          "Min batch: "      << options.minBatch                                                                        << std::endl <<
          "Opt batch: "      << options.optBatch                                                                        << std::endl <<
          "Max batch: "      << options.maxBatch                                                                        << std::endl <<
          "Workspace: "      << options.workspace << " MB"                                                              << std::endl <<
          "minTiming: "      << options.minTiming                                                                       << std::endl <<
          "avgTiming: "      << options.avgTiming                                                                       << std::endl <<
          "Precision: "      << (options.fp16 ? "FP16" : (options.int8 ? "INT8" : "FP32"))                              << std::endl <<
          "Calibration: "    << (options.int8 && options.calibration.empty() ? "Dynamic" : options.calibration.c_str()) << std::endl <<
          "Safe mode: "      << boolToEnabled(options.safe)                                                             << std::endl <<
          "Save engine: "    << (options.save ? options.engine : "")                                                    << std::endl <<
          "Load engine: "    << (options.load ? options.engine : "")                                                    << std::endl;
// clang-format on

    printIOFormats(os, "Input", options.inputFormats);
    printIOFormats(os, "Output", options.outputFormats);

    return os;
}

std::ostream& operator<<(std::ostream& os, const SystemOptions& options)
{
// clang-format off
    os << "=== System Options ==="                                                                << std::endl <<

          "Device: "  << options.device                                                           << std::endl <<
          "DLACore: " << (options.DLACore != -1 ? std::to_string(options.DLACore) : "")           <<
                         (options.DLACore != -1 && options.fallback ? "(With GPU fallback)" : "") << std::endl;
// clang-format on
    os << "Plugins:";
    for (const auto p : options.plugins)
    {
        os << " " << p;
    }
    os << std::endl;

    return os;
}

std::ostream& operator<<(std::ostream& os, const InferenceOptions& options)
{
// clang-format off
    os << "=== Inference Options ==="                                        << std::endl <<

          "Batch: "          << options.batch                                << std::endl <<
          "Iterations: "     << options.iterations << " (" << options.warmup <<
                                                      " ms warm up)"         << std::endl <<
          "Duration: "       << options.duration   << "s"                    << std::endl <<
          "Sleep time: "     << options.sleep      << "ms"                   << std::endl <<
          "Streams: "        << options.streams                              << std::endl <<
          "Multithreading: " << boolToEnabled(options.threads)               << std::endl <<
          "CUDA Graph: "     << boolToEnabled(options.graph)                 << std::endl <<
          "Skip inference: " << boolToEnabled(options.skip)                  << std::endl;
// clang-format on

    return os;
}

std::ostream& operator<<(std::ostream& os, const ReportingOptions& options)
{
// clang-format off
    os << "=== Reporting Options ==="                                       << std::endl <<

          "Verbose: "                     << boolToEnabled(options.verbose) << std::endl <<
          "Averages: "                    << options.avgs << " inferences"  << std::endl <<
          "Percentile: "                  << options.percentile             << std::endl <<
          "Dump output: "                 << boolToEnabled(options.output)  << std::endl <<
          "Profile: "                     << boolToEnabled(options.profile) << std::endl <<
          "Export timing to JSON file: "  << options.exportTimes            << std::endl <<
          "Export profile to JSON file: " << options.exportProfile          << std::endl;
// clang-format on

    return os;
}

std::ostream& operator<<(std::ostream& os, const AllOptions& options)
{
    os << options.model << options.build << options.system << options.inference << options.reporting << std::endl;
    return os;
}

void BaseModelOptions::help(std::ostream& os)
{
// clang-format off
    os << "  --uff=<file>                UFF model"                                             << std::endl <<
          "  --onnx=<file>               ONNX model"                                            << std::endl <<
          "  --model=<file>              Caffe model (default = no model, random weights used)" << std::endl;
// clang-format on
}

void UffInput::help(std::ostream& os)
{
// clang-format off
    os << "  --uffInput=<name>,X,Y,Z     Input blob name and its dimensions (X,Y,Z=C,H,W), it can be specified "
                                                       "multiple times; at least one is required for UFF models" << std::endl <<
          "  --uffNHWC                   Set if inputs are in the NHWC layout instead of NCHW (use "             <<
                                                                    "X,Y,Z=H,W,C order in --uffInput)"           << std::endl;
// clang-format on
}

void ModelOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Model Options ==="                                                                                 << std::endl;
    BaseModelOptions::help(os);
    os << "  --deploy=<file>             Caffe prototxt file"                                                     << std::endl <<
          "  --output=<name>[,<name>]*   Output names (it can be specified multiple times); at least one output "
                                                                                  "is required for UFF and Caffe" << std::endl;
    UffInput::help(os);
// clang-format on
}

void BuildOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Build Options ==="                                                                                                  << std::endl <<

          "  --minBatch=N                Set min batch size when building the engine (default = " << defaultMinBatch << ")"        << std::endl <<
          "  --optBatch=N                Set optimal batch size when building the engine (default = maxBatch size)"                << std::endl <<
          "  --maxBatch=N                Set max batch size when building the engine (default = minBatch size)"                    << std::endl <<
          "                              Note: batch ranges with more than one batch size are enabled by dynamic shapes"           << std::endl <<
          "  --inputIOFormats=spec       Type and formats of the input tensors (default = all inputs in fp32:chw)"                 << std::endl <<
          "  --outputIOFormats=spec      Type and formats of the output tensors (default = all outputs in fp32:chw)"               << std::endl <<
          "                              IO Formats: spec  ::= IOfmt[\",\"spec]"                                                   << std::endl <<
          "                                          IOfmt ::= type:fmt"                                                           << std::endl <<
          "                                          type  ::= \"fp32\"|\"fp16\"|\"int32\"|\"int8\""                               << std::endl <<
          "                                          fmt   ::= (\"chw\"|\"chw2\"|\"chw4\"|\"hwc8\"|\"chw16\"|\"chw32\")[\"+\"fmt]" << std::endl <<
          "  --workspace=N               Set workspace size in megabytes (default = " << defaultWorkspace << ")"                   << std::endl <<
          "  --minTiming=M               Set the minimum number of iterations used in kernel "
                                                                "selection (default = " << defaultMinTiming << ")"                 << std::endl <<
          "  --avgTiming=M               Set the number of times averaged in each iteration for "
                                                                   "kernel selection (default = " << defaultAvgTiming << ")"       << std::endl <<
          "  --fp16                      Enable fp16 mode (default = disabled)"                                                    << std::endl <<
          "  --int8                      Run in int8 mode (default = disabled)"                                                    << std::endl <<
          "  --calib=<file>              Read INT8 calibration cache file"                                                         << std::endl <<
          "  --safe                      Only test the functionality available in safety restricted flows"                         << std::endl <<
          "  --saveEngine=<file>         Save the serialized engine"                                                               << std::endl <<
          "  --loadEngine=<file>         Load a serialized engine"                                                                 << std::endl;
// clang-format on
}

void SystemOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== System Options ==="                                                                         << std::endl <<
          "  --device=N                  Select cuda device N (default = " << defaultDevice << ")"         << std::endl <<
          "  --useDLACore=N              Select DLA core N for layers that support DLA (default = none)"   << std::endl <<
          "  --allowGPUFallback          When DLA is enabled, allow GPU fallback for unsupported layers "
                                                                                    "(default = disabled)" << std::endl;
    os << "  --plugins                   Plugin library (.so) to load (can be specified multiple times)"   << std::endl;
// clang-format on
}

void InferenceOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Inference Options ==="                                                                                 << std::endl <<
          "  --batch=N                   Set batch size (default = " << defaultBatch << ")"                           << std::endl <<
          "  --iterations=N              Run at least N inference iterations (default = " << defaultIterations << ")" << std::endl <<
          "  --warmUp=N                  Run for N milliseconds to warmup before measuring performance "
                                                                       "(default = " << defaultWarmUp << ")"          << std::endl <<
          "  --duration=N                Run performance measurements for at least N seconds "
                                                      "wallclock time (default = " << defaultDuration << ")"          << std::endl <<
          "  --sleepTime=N               Delay inference start with a gap of N milliseconds between "
                                                     "launch and compute (default = " << defaultSleep << ")"          << std::endl <<
          "  --streams=N                 Instantiate N engines to use concurrently (default = " <<
                                                                                       defaultStreams << ")"          << std::endl <<
          "  --threads                   Enable multithreading to drive engines with independent "
                                                                              "threads (default = disabled)"          << std::endl <<
          "  --useCudaGraph              Use cuda graph to capture engine execution and then launch "
                                                                               "inference (default = false)"          << std::endl <<
          "  --buildOnly                 Skip inference perf measurement (default = disabled)"                        << std::endl;
// clang-format on
}

void ReportingOptions::help(std::ostream& os)
{
// clang-format off
    os << "=== Reporting Options ==="                                                                    << std::endl <<
          "  --verbose                   Use verbose logging (default = false)"                          << std::endl <<
          "  --avgRuns=N                 Report performance measurements averaged over N consecutive "
                                                       "iterations (default = " << defaultAvgRuns << ")" << std::endl <<
          "  --percentile=P              Report performance for the P percentage (0<=P<=100, 0 "
                                        "representing max perf, and 100 representing min perf; (default"
                                                                      " = " << defaultPercentile << "%)" << std::endl <<
          "  --dumpOutput                Print the output tensor(s) of the last inference iteration "
                                                                                  "(default = disabled)" << std::endl <<
          "  --dumpProfile               Print profile information per layer (default = disabled)"       << std::endl <<
          "  --exportTimes=<file>        Write the timing results in a json file (default = disabled)"   << std::endl <<
          "  --exportProfile=<file>      Write the profile information per layer in a json file "
                                                                              "(default = disabled)" << std::endl;
// clang-format on
}

void helpHelp(std::ostream& os)
{
    os << "=== Help ==="                                     << std::endl <<
          "  --help                      Print this message" << std::endl;
}

void AllOptions::help(std::ostream& os)
{
    ModelOptions::help(os);
    os << std::endl;
    BuildOptions::help(os);
    os << std::endl;
    InferenceOptions::help(os);
    os << "Note: if a batch size is specified only for inference, it will be used also as min, opt, and max batch size for the builder" << std::endl;
    os << std::endl;
    ReportingOptions::help(os);
    os << std::endl;
    SystemOptions::help(os);
    os << std::endl;
    helpHelp(os);
}

} // namespace sample
