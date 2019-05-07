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

#ifndef BATCH_STREAM_H
#define BATCH_STREAM_H

#include "NvInfer.h"
#include "common.h"
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>

class BatchStream
{
public:
    BatchStream(
        int batchSize, int maxBatches, std::string prefix, std::string suffix, std::vector<std::string> directories)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mPrefix(prefix)
        , mSuffix(suffix)
        , mDataDir(directories)
    {
        FILE* file = fopen(locateFile(mPrefix + std::string("0") + mSuffix, mDataDir).c_str(), "rb");
        assert(file != nullptr);
        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        mDims.nbDims = 4;  // The number of dimensions.
        mDims.d[0] = d[0]; // Batch Size
        mDims.d[1] = d[1]; // Channels
        mDims.d[2] = d[2]; // Height
        mDims.d[3] = d[3]; // Width
        assert(mDims.d[0] > 0 && mDims.d[1] > 0 && mDims.d[2] > 0 && mDims.d[3] > 0);
        fclose(file);

        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    BatchStream(int batchSize, int maxBatches, std::string prefix, std::vector<std::string> directories)
        : BatchStream(batchSize, maxBatches, prefix, ".batch", directories)
    {
    }

    BatchStream(
        int batchSize, int maxBatches, nvinfer1::Dims dims, std::string listFile, std::vector<std::string> directories)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mDims(dims)
        , mListFile(listFile)
        , mDataDir(directories)
    {
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    // Resets data members
    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next()
    {
        if (mBatchCount == mMaxBatches)
        {
            return false;
        }

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
            if (mFileBatchPos == mDims.d[0] && !update())
            {
                return false;
            }

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(
                getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
            std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
        }
        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount)
    {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
        {
            next();
        }
        mBatchCount = x;
    }

    float* getBatch() { return mBatch.data(); }

    float* getLabels() { return mLabels.data(); }

    int getBatchesRead() const { return mBatchCount; }

    int getBatchSize() const { return mBatchSize; }

    nvinfer1::Dims getDims() const { return mDims; }

private:
    float* getFileBatch() { return mFileBatch.data(); }

    float* getFileLabels() { return mFileLabels.data(); }

    bool update()
    {
        if (mListFile.empty())
        {
            std::string inputFileName = locateFile(mPrefix + std::to_string(mFileCount++) + mSuffix, mDataDir);
            FILE* file = fopen(inputFileName.c_str(), "rb");
            if (!file)
            {
                return false;
            }

            int d[4];
            size_t readSize = fread(d, sizeof(int), 4, file);
            assert(readSize == 4);
            assert(mDims.d[0] == d[0] && mDims.d[1] == d[1] && mDims.d[2] == d[2] && mDims.d[3] == d[3]);
            size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
            assert(readInputCount == size_t(mDims.d[0] * mImageSize));
            size_t readLabelCount = fread(getFileLabels(), sizeof(float), mDims.d[0], file);
            assert(readLabelCount == 0 || readLabelCount == size_t(mDims.d[0]));

            fclose(file);
        }
        else
        {
            std::vector<std::string> fNames;
            std::ifstream file(locateFile(mListFile, mDataDir), std::ios::binary);
            if (!file)
            {
                return false;
            }

            gLogInfo << "Batch #" << mFileCount << std::endl;
            file.seekg(((mBatchCount * mBatchSize)) * 7);

            for (int i = 1; i <= mBatchSize; i++)
            {
                std::string sName;
                std::getline(file, sName);
                sName = sName + ".ppm";
                gLogInfo << "Calibrating with file " << sName << std::endl;
                fNames.emplace_back(sName);
            }

            mFileCount++;

            const int imageC = 3;
            const int imageH = 300;
            const int imageW = 300;
            std::vector<samplesCommon::PPM<imageC, imageH, imageW>> ppms(fNames.size());
            for (uint32_t i = 0; i < fNames.size(); ++i)
            {
                readPPMFile(locateFile(fNames[i], mDataDir), ppms[i]);
            }

            std::vector<float> data(samplesCommon::volume(mDims));
            const float scale = 2.0 / 255.0;
            const float bias = 1.0;
            long int volChl = mDims.d[2] * mDims.d[3];

            // Normalize input data
            for (int i = 0, volImg = mDims.d[1] * mDims.d[2] * mDims.d[3]; i < mBatchSize; ++i)
            {
                for (int c = 0; c < mDims.d[1]; ++c)
                {
                    for (int j = 0; j < volChl; ++j)
                    {
                        data[i * volImg + c * volChl + j] = scale * float(ppms[i].buffer[j * mDims.d[1] + c]) - bias;
                    }
                }
            }

            std::copy_n(data.data(), mDims.d[0] * mImageSize, getFileBatch());
        }

        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    std::vector<float> mBatch;         //!< Data for the batch
    std::vector<float> mLabels;        //!< Labels for the batch
    std::vector<float> mFileBatch;     //!< List of image files
    std::vector<float> mFileLabels;    //!< List of label files
    std::string mPrefix;               //!< Batch file name prefix
    std::string mSuffix;               //!< Batch file name suffix
    nvinfer1::Dims mDims;              //!< Input dimensions
    std::string mListFile;             //!< File name of the list of image names
    std::vector<std::string> mDataDir; //!< Directories where the files can be found
};

#endif
