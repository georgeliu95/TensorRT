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

namespace RNNDataUtil
{
// C++ adaptation of data_utils.py from https://github.com/rafaljozefowicz/lm
const uint32_t none = -1;
class Vocabulary
{
    std::map<std::string, uint32_t> mTokenToId;
    std::map<std::string, uint32_t> mTokenToCount;
    std::vector<std::string> mIdToToken;
    uint32_t mNumTokens;
    uint32_t mSId;
    uint32_t mUnkId;
    std::string mS;
    std::string mU;

public:
    Vocabulary()
        : mNumTokens(0)
        , mSId(none)
        , mUnkId(none)
        , mS("<S>")
        , mU("<UNK>"){};
    uint32_t getNumTokens() const { return mNumTokens; };
    uint32_t getSId() const { return mSId; }
    std::string getS() const { return mS; };
    std::string getU() const { return mU; };
    void add(const std::string& token, uint32_t count);
    uint32_t getId(const std::string& token);
    std::string getToken(uint32_t id) const;
    void finalize();
    void fromFile(const std::string& fileName);
};

int testVocabulary(Vocabulary& vocabulary);

class Dataset
{
    // This class should produce identical results to the python script

    // contains word token ids of all parsed lines (numLines, numTokensPerLine)
    std::vector<std::vector<uint32_t>> mAllTokens;
    // global line id for iterateOnce
    int32_t mLineId;
    uint32_t mBatchSize;
    uint32_t mSeqSize;
    Vocabulary* mVocabularyPtr;
    // size batchSize arrays to track progress from multiple iterations
    std::vector<uint32_t> mLinePos;
    std::vector<uint32_t> mLineIds;
    std::vector<uint32_t> mDist;

public:
    Dataset(uint32_t batchSize, uint32_t seqSize, Vocabulary& vocabulary);
    void parseFile(const std::string& fileName);
    // x and y batchSize * seqSize Hot vectors. seqSize - fast dimension
    void iterateOnce(std::vector<uint32_t>& x, std::vector<uint32_t>& y);
    Vocabulary* getVocabularyPtr() { return mVocabularyPtr; };
};

void getHotFromData(std::vector<uint32_t>& x, std::vector<uint32_t>& y, uint32_t batchSize, uint32_t seqSize, const Vocabulary& vocabulary);
} // end namespace RNNDataUtil
