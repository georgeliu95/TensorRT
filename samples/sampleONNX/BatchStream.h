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

#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "NvInfer.h"

//...Filed https://jirasw.nvidia.com/browse/TRT-4616
//...This is unnecessary complicated!

const char* gNetworkName = "ONNX";

struct PPM2
{
    std::string magic, fileName;
    int h, w, max;
    uint8_t buffer[3 * 300 * 300];
};

std::string locateFile(const std::string& input);

void readPPMFileBatch(const std::string& filename, PPM2& ppm)
{
    ppm.fileName = filename;
    std::ifstream infile(locateFile(filename), std::ifstream::binary);
    infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

class BatchStream
{
    int mBatchSize;
    int mMaxBatches;
    int mBatchCount;

    int mFileCount;
    int mFileBatchPos;
    int mImageSize;

    nvinfer1::DimsNCHW mDims;
    std::vector<float> mBatch;
    std::vector<float> mLabels;
    std::vector<float> mFileBatch;
    std::vector<float> mFileLabels;

    string mCalibrationList;
    std::ifstream mCalibrationListFile;

public:
    BatchStream(int batchSize, int maxBatches, const string CalibrationList)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mBatchCount(0)
        , mFileCount(0)
        , mFileBatchPos(0)
        , mCalibrationList(CalibrationList)
        , mCalibrationListFile(mCalibrationList)
    {

        if (!mCalibrationListFile.is_open())
        {
            string msg("Failed to open the calibration list file ");
            msg += CalibrationList;
            throw std::runtime_error(msg);
        }

        mDims = nvinfer1::DimsNCHW{batchSize, 3, 300, 300};
        mImageSize = mDims.c() * mDims.h() * mDims.w();
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.n() * mImageSize, 0);
        mFileLabels.resize(mDims.n(), 0);
        reset(0);
    }

    BatchStream(const BatchStream& src)
        : BatchStream(src.getBatchSize(), src.getMaxBatches(), src.getCalibrationList())
    {
    }

    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.n();
        skip(firstBatch);
    }

    bool next()
    {
        if (mBatchCount == mMaxBatches)
            return false;

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.n());
            if (mFileBatchPos == mDims.n() && !update())
                return false;

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.n() - mFileBatchPos);
            std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
            //std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
        }
        mBatchCount++;
        return true;
    }

    void skip(int skipCount)
    {
        if (mBatchSize >= mDims.n() && mBatchSize % mDims.n() == 0 && mFileBatchPos == mDims.n())
        {
            mFileCount += skipCount * mBatchSize / mDims.n();
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
            next();
        mBatchCount = x;
    }

    float* getBatch() { return &mBatch[0]; }
    float* getLabels() { return &mLabels[0]; }
    int getBatchesRead() const { return mBatchCount; }
    int getBatchSize() const { return mBatchSize; }
    int getMaxBatches() const { return mMaxBatches; }
    nvinfer1::DimsNCHW getDims() const { return mDims; }
    string getCalibrationList() const { return mCalibrationList; }

private:
    float* getFileBatch() { return &mFileBatch[0]; }
    float* getFileLabels() { return &mFileLabels[0]; }

    bool update()
    {
        std::vector<std::string> fNames;

        if (mCalibrationListFile)
        {
            mCalibrationListFile.seekg(((mBatchCount * mBatchSize)) * 7);
        }
        for (int i = 1; i <= mBatchSize; i++)
        {
            std::string sName;
            std::getline(mCalibrationListFile, sName);
            sName = sName + ".ppm";

            cout << " Using Calibration File " << sName << std::endl;
            fNames.emplace_back(sName);
        }
        mFileCount++;

        std::vector<PPM2> ppms(fNames.size());
        for (uint32_t i = 0; i < fNames.size(); ++i)
        {
            readPPMFileBatch(fNames[i], ppms[i]);
        }
        vector<float> data(mBatchSize * mDims.c() * mDims.h() * mDims.w());

        for (int i = 0, volImg = mDims.c() * mDims.h() * mDims.w(); i < mBatchSize; ++i)
        {
            for (int c = 0; c < mDims.c(); ++c)
            {
                for (unsigned j = 0, volChl = mDims.h() * mDims.w(); j < volChl; ++j)
                {
                    data[i * volImg + c * volChl + j] = (2.0 / 255.0) * float(ppms[i].buffer[j * mDims.c() + c]) - 1.0;
                }
            }
        }

        std::copy_n(data.data(), mDims.n() * mImageSize, getFileBatch());
        mFileBatchPos = 0;
        return true;
    }
};

#endif
