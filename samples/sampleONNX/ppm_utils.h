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

#ifndef PARSER_ONNX_H
#define PARSER_ONNX_H

#include <iostream>
#include <iterator>
#include <math.h>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "logger.h"

using namespace std;

namespace PPM
{

struct PPM
{
    std::string magic, fileName;
    int c, h, w, max;
    uint8_t* _buffer = NULL;

    PPM(int channel, int height, int width)
    {
        c = channel;
        h = height;
        w = width;
    }

    void reserve_buffer()
    {
        gLogInfo << " PPM::reserve_buffer(): Allocating buffer C:" << c << "\tH:" << h << "\tW:" << w << "\t total size = " << c * w * h << std::endl;
        _buffer = new uint8_t[c * w * h];
    }
    ~PPM()
    {
        delete _buffer;
    }
};

struct BBox
{
    float x1, y1, x2, y2;
};

// simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, PPM& ppm)
{
    ppm.fileName = filename;
    std::ifstream infile(filename);
    infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
    if (ppm.max >= 256)
    {
        std::string message = "ERROR: PPM files with max value exceeding 255 currently is not supported";
        throw std::range_error(message);
    }
    infile.seekg(1, infile.cur);
    ppm.reserve_buffer();
    infile.read(reinterpret_cast<char*>(ppm._buffer), ppm.w * ppm.h * ppm.c);
    infile.close();
}

nvinfer1::Dims getDims(const std::string& filename)
{
    //...The PPM file is the raster for Height rows, each row consits of widts pixels
    //...Each pixel is the triplet of red, green and blue samples.
    //...Each sample is represented by a binary by either 1 (if max < 256) or 2 bytes
    //...max is a number 1 < 65536
    std::ifstream infile(filename);
    std::string magic;
    int c = 3, h, w, max;

    infile >> magic >> w >> h >> max;
    if (magic != "P6")
    {
        std::string message = "ERROR: The file does not appear to be a PPM file.";
        throw std::range_error(message);
    }
    if (max >= 256)
    {
        std::string message = "ERROR: PPM files with max value exceeding 255 currently is not supported";
        throw std::range_error(message);
    }
    infile.close();
    return nvinfer1::DimsCHW{c, h, w};
}

template <typename T>
struct Image
{
    int width, height, nchan;
    std::vector<T> data;
    int getDataSize() const { return width * height * nchan; }
    Image() {}
    //	Image(const Image& src) : width(src.width), height(src.height)
    Image(const PPM& ppm)
        : width(ppm.w)
        , height(ppm.h)
        , nchan(ppm.c)
        , data(ppm._buffer, ppm._buffer + ppm.w * ppm.h * ppm.c)
    {
    }
    Image(const vector<T> inp, int w, int h, int c)
        : width(w)
        , height(h)
        , nchan(c)
        , data(checkSize(inp))
    {
    }

    const std::vector<T>& checkSize(const vector<T>& inp) const
    {
        if (getDataSize() != static_cast<int>(inp.size()))
        {
            string msg = " Error in Image::Image(const vector<T> inp, int w, int h, int c) : inp.size() != w*h*c "
                + to_string(inp.size()) + " w: " + to_string(width) + " h: " + to_string(height) + " c: " + to_string(nchan);
            throw std::out_of_range(msg);
        }
        return inp;
    }
};

// Crappy nearest-neighbour resize
template <typename T>
Image<T> resize_image(Image<T> const& image, int new_height, int new_width)
{
    Image<T> new_image;
    new_image.height = new_height;
    new_image.width = new_width;
    new_image.nchan = image.nchan;
    new_image.data.resize(new_height * new_width * image.nchan);
    for (int y = 0; y < new_image.height; ++y)
    {
        for (int x = 0; x < new_image.width; ++x)
        {
            for (int c = 0; c < new_image.nchan; ++c)
            {
                int dst_idx = (y * new_image.width + x) * new_image.nchan + c;
                int sy = rintf(y * image.height / float(new_image.height));
                int sx = rintf(x * image.width / float(new_image.width));
                int src_idx = (sy * image.width + sx) * image.nchan + c;
                new_image.data[dst_idx] = image.data[src_idx];
            }
        }
    }
    return new_image;
}

template <typename T>
Image<T> hwc2chw(Image<T> const& image)
{
    Image<T> new_image;
    new_image.height = image.height;
    new_image.width = image.width;
    new_image.nchan = image.nchan;
    new_image.data.resize(image.height * image.width * image.nchan);
    for (int c = 0; c < new_image.nchan; ++c)
    {
        for (int y = 0; y < new_image.height; ++y)
        {
            for (int x = 0; x < new_image.width; ++x)
            {
                int dst_idx = (c * image.height + y) * image.width + x;
                int src_idx = (y * image.width + x) * image.nchan + c;
                new_image.data[dst_idx] = image.data[src_idx];
            }
        }
    }
    return new_image;
}

//...LG this is a terrible hack!! data is resized to accomodate the required number of floating numbers.
template <typename T>
Image<T> convert_to_float(Image<T> const& image)
{
    Image<T> new_image;
    new_image.height = image.height;
    new_image.width = image.width;
    new_image.nchan = image.nchan;
    new_image.data.resize(image.height * image.width * image.nchan * sizeof(float));
    for (int i = 0; i < image.height * image.width * image.nchan; ++i)
    {
        //((float*)new_image.data.data())[i] = image.data[i] / 127.5f - 1;
        // ~PyTorch input normalization
        ((float*) new_image.data.data())[i] = (image.data[i] / 255.f - 0.45) / 0.225;
    }
    return new_image;
}

} // end namespace PPM

#endif
