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

#pragma once
#include <map>
#include <vector>

namespace RNNDataUtil
{
struct Params
{
    std::map<std::string, std::vector<std::string>> expected = {
        {"biglstm2x20x8.wts", {"<S>(2)", "store(1389)", "'(76)"}},
        {"biglstm2x20.wts", {"<S>(2)", "new(83)", ",(1)"}},
        {"toybiglstm2x20.wts", {"<S>(2)", "investment(679)", ",(1)"}},
    };
    //"toybiglstm2x20.wts"
    //"biglstm2x20.wts"
    std::string dataDir{"data/samples/big-lstm/"}, weightsFile{"toybiglstm2x20.wts"}, inputsFile{"news.en.heldout-00000-of-00050"},
        vocabFile{"1b_word_vocab.txt"};
    int device{0}, workspaceSize{31}, avgRuns{10}, useDLACore{-1};
    uint32_t layers{2}, seqSize{20}, projectedSize{128}, hiddenSize{projectedSize}, dataSize{projectedSize}, cellSize{512};
    uint32_t batchSize{1}, minibatchSize{batchSize}, iterations{12};
    bool half2{false}, int8{false}, verbose{false}, sm{false}, perplexity{false};
    float overrideGPUClocks{0.0F}, overrideMemClocks{0.0F}, overrideAchievableMemoryBandwidthRatio{0.0F};
};

void printUsage(Params& params);

bool parseArgs(int argc, char* argv[], Params& params);
} // end namespace RNNDataUtil
