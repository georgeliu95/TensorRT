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

#include <string>
#include <cstring>
#include <iostream>
#include "error_util.h"
#include "logger.h"
#include "params.h"

namespace RNNDataUtil
{
bool parseString(const char* arg, const char* name, std::string& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = arg + n + 3;
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool parseString(const char* arg, const char* name)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
    return match;
}

bool parseInt(const char* arg, const char* name, int& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atoi(arg + n + 3);
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool parseInt(const char* arg, const char* name, uint32_t& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = (uint32_t) atoi(arg + n + 3);
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool parseFloat(const char* arg, const char* name, float& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n) && arg[n + 2] == '=';
    if (match)
    {
        value = atof(arg + n + 3);
        gLogInfo << name << ": " << value << std::endl;
    }
    return match;
}

bool parseBool(const char* arg, const char* name, bool& value)
{
    size_t n = strlen(name);
    bool match = arg[0] == '-' && arg[1] == '-' && !strncmp(arg + 2, name, n);
    if (match)
    {
        gLogInfo << name << std::endl;
        value = true;
    }
    return match;
}

void printUsage(Params& rnnParams)
{
    printf("\n");
    printf("\nOptional params:\n");
    printf("  --datadir=<path>        Path string to match up to 10 levels from the current dir (default = %s)\n", rnnParams.dataDir.c_str());
    printf("  --weights=<file>        Name of the file with weights, located in the datadir (default = %s)\n", rnnParams.weightsFile.c_str());
    printf("  --batch=N               Set batch size (default = %d)\n", rnnParams.batchSize);
    printf("  --num_layers=N          Set number of LSTMP layers (default = %d)\n", rnnParams.layers);
    printf("  --num_steps=N           Set number of LSTMP time steps (default = %d)\n", rnnParams.seqSize);
    printf("  --state_size=N          Set number of LSTM memory cells (default = %d)\n", rnnParams.cellSize);
    printf("  --projected_size=N      Set size of LSTMP projection layer (default = %d)\n", rnnParams.projectedSize);
    printf("  --device=N              Set cuda device to N (default = %d)\n", rnnParams.device);
    printf("  --iterations=N          Run N iterations (default = %d)\n", rnnParams.iterations);
    printf("  --avgRuns=N             Set avgRuns to N - perf is measured as an average of avgRuns (default=%d)\n", rnnParams.avgRuns);
    printf("  --workspace=N           Set workspace size in megabytes (default = %d)\n", rnnParams.workspaceSize);
    printf("  --half2                 Run in paired fp16 mode (default = false)\n");
    printf("  --int8                  Run in int8 mode (default = false)\n");
    printf("  --verbose               Use verbose logging (default = false)\n");
    printf("  --sm                    Disable output Softmax layer (default = false)\n");
    printf("  --overrideGPUClocks=val Override GPU clocks for %%SOL calculation, in GHz\n");
    printf("  --overrideMemClocks=val Override memory clocks for %%SOL calculation, in GHz\n");
    printf("  --overrideBWRatio=val   Override practical/theoretical peak bandwidth ratio for %%SOL calculation\n");
    printf("  --useDLACore            Specify a DLA core for layers that support DLA. Value can range from 0 to N-1, where N is the number of DLA cores on the platform\n");

    fflush(stdout);
}

bool parseArgs(int argc, char* argv[], Params& params)
{
    if (argc < 1) // no mandatory parameters currenly
    {
        return false;
    }

    for (int j = 1; j < argc; j++)
    {

        if (parseString(argv[j], "usage") || parseString(argv[j], "help"))
            return false;

        if (parseString(argv[j], "datadir", params.dataDir) || parseString(argv[j], "weights", params.weightsFile) || parseString(argv[j], "input", params.inputsFile) || parseString(argv[j], "vocabulary", params.vocabFile))
            continue;

        if (parseInt(argv[j], "iterations", params.iterations) || parseInt(argv[j], "avgRuns", params.avgRuns)
            || parseInt(argv[j], "device", params.device) || parseInt(argv[j], "workspace", params.workspaceSize))
            continue;

        if (parseInt(argv[j], "batch", params.batchSize))
        {
            params.minibatchSize = params.batchSize;
            continue;
        }

        if (parseInt(argv[j], "num_layers", params.layers) || parseInt(argv[j], "num_steps", params.seqSize) || parseInt(argv[j], "state_size", params.cellSize))
            continue;

        if (parseInt(argv[j], "projected_size", params.projectedSize))
        {
            params.dataSize = params.projectedSize;
            params.hiddenSize = params.projectedSize;
            continue;
        }

        if (parseBool(argv[j], "half2", params.half2) || parseBool(argv[j], "int8", params.int8) || parseBool(argv[j], "verbose", params.verbose) || parseBool(argv[j], "sm", params.sm)
            || parseBool(argv[j], "perplexity", params.perplexity)
            || parseInt(argv[j], "useDLACore", params.useDLACore))
        {
            if (params.perplexity)
            {
                params.sm = true;
            }
            continue;
        }

        if (parseFloat(argv[j], "overrideGPUClocks", params.overrideGPUClocks) || parseFloat(argv[j], "overrideMemClocks", params.overrideMemClocks)
            || parseFloat(argv[j], "overrideBWRatio", params.overrideAchievableMemoryBandwidthRatio))
        {
            continue;
        }

        printf("Unknown argument: %s\n", argv[j]);
        return false;
    }

    if (params.int8)
    {
        FatalError("Int8 is supported");
    }

    return true;
}

} // end namespace RNNDataUtil
