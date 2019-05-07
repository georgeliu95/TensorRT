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

#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include "NvInfer.h"
#include "BatchStream.h"

//! \class EntropyCalibratorImpl
//! 
//! \brief Implements common functionality for Entropy calibrators.
//!
class EntropyCalibratorImpl
{
public:
    EntropyCalibratorImpl(BatchStream& stream, int firstBatch, std::string networkName, const char* inputBlobName, bool readCache = true)
        : mStream(stream)
        , mCalibrationTableName("CalibrationTable"+networkName)
        , mInputBlobName(inputBlobName)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }
    
    virtual ~EntropyCalibratorImpl()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const { return mStream.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings)
    {
        if (!mStream.next())
        {
            return false;
        }
        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], mInputBlobName));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length)
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    virtual void writeCalibrationCache(const void* cache, size_t length)
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    BatchStream mStream;
    void* mDeviceInput{nullptr};
    std::string mCalibrationTableName;
    size_t mInputCount;
    const char* mInputBlobName;
    bool mReadCache{true};
    std::vector<char> mCalibrationCache;
};

//! \class Int8EntropyCalibrator2
//! 
//! \brief Implements Entropy calibrator 2. 
//!  CalibrationAlgoType is kENTROPY_CALIBRATION_2.
//!
class Int8EntropyCalibrator2 : public IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(BatchStream& stream, int firstBatch, const char* networkName, const char* inputBlobName, bool readCache = true)
        : mImpl(stream, firstBatch, networkName, inputBlobName, readCache)
    {
    }
    
    int getBatchSize() const override { return mImpl.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        return mImpl.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) override
    {
        return mImpl.readCalibrationCache(length);
    }
    
    void writeCalibrationCache(const void* cache, size_t length) override
    {
        mImpl.writeCalibrationCache(cache, length);
    }
private:
    EntropyCalibratorImpl mImpl;
};

class Int8LegacyCalibrator : public nvinfer1::IInt8LegacyCalibrator
{
public:
    Int8LegacyCalibrator(BatchStream& stream, int firstBatch, double cutoff, double quantile, std::string networkName, bool readCache = true)
        : mStream(stream)
        , mFirstBatch(firstBatch)
        , mCalibrationTableName("CalibrationTable"+networkName)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        reset(cutoff, quantile);
    }

    virtual ~Int8LegacyCalibrator()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }
    double getQuantile() const override { return mQuantile; }
    double getRegressionCutoff() const override { return mCutoff; }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
            return false;

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(mCalibrationTableName, std::ios::binary);
        input >> std::noskipws;

        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(mCalibrationTableName, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

    const void* readHistogramCache(size_t& length) override
    {
        length = mHistogramCache.size();
        return length ? &mHistogramCache[0] : nullptr;
    }

    void writeHistogramCache(const void* cache, size_t length) override
    {
        mHistogramCache.clear();
        std::copy_n(reinterpret_cast<const char*>(cache), length, std::back_inserter(mHistogramCache));
    }

    void reset(double cutoff, double quantile)
    {
        mCutoff = cutoff;
        mQuantile = quantile;
        mStream.reset(mFirstBatch);
    }

private:
    BatchStream mStream;
    int mFirstBatch;
    double mCutoff, mQuantile;
    std::string mCalibrationTableName;
    bool mReadCache{true};

    size_t mInputCount;
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache, mHistogramCache;
};
#endif // CALIBRATOR_H
