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

#include <algorithm>
#include <cctype>
#include <clocale>
#include <fstream>
#include <functional>
#include <iostream>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <assert.h>

#include "logger.h"
#include "data_utils.h"

namespace RNNDataUtil
{

#define DATA_LOCALE "en_US.UTF-8"
// trim from stackoverflow
// trim from start
static inline std::string& ltrim(std::string& s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
    return s;
}

// trim from end
static inline std::string& rtrim(std::string& s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         [](unsigned char ch) {
                             return !std::isspace(ch);
                         })
                .base(),
            s.end());
    return s;
}

// trim from both ends
static inline void trim(std::string& s)
{
    ltrim(rtrim(s));
}

void Vocabulary::add(const std::string& token, uint32_t count)
{
    mTokenToId[token] = mNumTokens;
    mTokenToCount[token] = count;
    mIdToToken.push_back(token);
    mNumTokens += 1;
}
uint32_t Vocabulary::getId(const std::string& token)
{
    std::map<std::string, uint32_t>::iterator it;
    it = mTokenToId.find(token);
    if (it != mTokenToId.end())
        return it->second;
    return mUnkId;
}
std::string Vocabulary::getToken(uint32_t id) const
{
    if (id < mIdToToken.size())
        return mIdToToken[id];
    else
    {
        gLogError << "getToken: out of bound access" << std::endl;
        exit(-1);
    }
}
void Vocabulary::finalize()
{
    mSId = getId(mS);
    mUnkId = getId(mU);
}
void Vocabulary::fromFile(const std::string& fileName)
{
    std::setlocale(LC_ALL, DATA_LOCALE);
    // file should contain "<S>" and "<UNK>" tokens
    std::string line;
    std::ifstream inFile(fileName.c_str());
    std::string word;
    uint32_t count;
    while (std::getline(inFile, line))
    {
        trim(line);
        std::istringstream iss(line);
        iss >> word >> count;
        add(word, count);
    }
    finalize();
}

int testVocabulary(Vocabulary& vocabulary)
{
    //testing 1b_word_vocab.txt
    assert(vocabulary.getId("<S>") == 2);
    assert(vocabulary.getId("<UNK>") == 38);
    assert(vocabulary.getNumTokens() == 793470);
    assert(vocabulary.getToken(78423) == "Perminov");
    gLogInfo << "Vocabulary test passed" << std::endl;
    return 0;
}

Dataset::Dataset(uint32_t batchSize, uint32_t seqSize, Vocabulary& vocabulary)
    : mLineId(-1)
    , mBatchSize(batchSize)
    , mSeqSize(seqSize)
    , mVocabularyPtr(&vocabulary)
{
    mLinePos.resize(batchSize, none);
    mLineIds.resize(batchSize, 0);
    mDist.resize(batchSize, 0);
}

void Dataset::parseFile(const std::string& fileName)
{
    std::setlocale(LC_ALL, DATA_LOCALE);
    std::string line;
    std::ifstream inFile(fileName.c_str());
    int lineId = 0;
    while (std::getline(inFile, line))
    {
        mAllTokens.push_back(std::vector<uint32_t>());
        mAllTokens[lineId].push_back(mVocabularyPtr->getSId());
        trim(line);
        std::istringstream iss(line);
        std::string buf;
        while (iss >> buf)
        {
            mAllTokens[lineId].push_back(mVocabularyPtr->getId(buf));
        }
        mAllTokens[lineId].push_back(mVocabularyPtr->getSId());
        lineId++;
    }
}

void Dataset::iterateOnce(std::vector<uint32_t>& x, std::vector<uint32_t>& y)
{
    x.resize(mSeqSize * mBatchSize);
    y.resize(mSeqSize * mBatchSize);
    for (uint32_t n = 0; n < mBatchSize; n++)
    {
        uint32_t tokensFilled = 0;
        while (tokensFilled < mSeqSize)
        {
            if (mLinePos[n] == none || mDist[n] <= 1)
            {
                mLineId++;
                if ((uint32_t) mLineId >= mAllTokens.size())
                {
                    gLogError << "Amount of data provided is not sufficient for batch or time sequence size requested" << std::endl;
                    exit(-1);
                }
                mLinePos[n] = 0;
                mLineIds[n] = mLineId;
            }
            uint32_t curLineId = mLineIds[n];
            uint32_t curLinePos = mLinePos[n];
            mDist[n] = mAllTokens[curLineId].size() - curLinePos;
            uint32_t numTokens = std::min(mDist[n] - 1, mSeqSize - tokensFilled);
            std::copy(&mAllTokens[curLineId][curLinePos], &mAllTokens[curLineId][curLinePos + numTokens], &x[n * mSeqSize + tokensFilled]);

            // Add bounds check for Windows.
            if ((curLinePos + 1 + numTokens) < mAllTokens[curLineId].size())
            {
                std::copy(&mAllTokens[curLineId][curLinePos + 1], &mAllTokens[curLineId][curLinePos + 1 + numTokens], &y[n * mSeqSize + tokensFilled]);
            }
            mLinePos[n] += numTokens;
            tokensFilled += numTokens;
        }
    }
}
#undef DATA_LOCALE
} // end namespace RNNDataUtil
