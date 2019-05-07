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

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

//Num of lReLU layers in the YOLO V2 network
#define NUM_RELU_IN_YOLO_V2 22

softmaxTree* gSmTree;
int gOutputClsSize; // number of labelled classes
int gNumBoundBox;   // each grid cell predicts 'gNumBoundBox' bounding boxes

// lReLU parameter
static const float kNEG_SLOPEV2 = 0.1f;

enum PluginLayerType : int
{
    kLEAKY,
    kREORG,
    kREGION,
    kINVALID
};

bool matchStringv2(const char* pattern, const char* name)
{
    for (int i = 0; pattern[i] != '\0'; i++)
    {
        if (name[i] != pattern[i])
            return false;
    }
    return true;
}

PluginLayerType getPluginTypev2(const char* layerName)
{
    if (matchStringv2("leaky", layerName))
        return PluginLayerType::kLEAKY;
    if (matchStringv2("reorg", layerName))
        return PluginLayerType::kREORG;
    if (matchStringv2("region", layerName))
        return PluginLayerType::kREGION;
    return PluginLayerType::kINVALID;
}

class YOLOv2PluginFactory : public nvcaffeparser1::IPluginFactoryV2
{
public:
    YOLOv2PluginFactory()
        : mPluginReorg(nullptr, pluginDeleter)
        , mPluginRegion(nullptr, pluginDeleter)
    {
        for (unsigned i = 0; i < NUM_RELU_IN_YOLO_V2; i++)
        {
            mPluginLRelu.push_back({nullptr, pluginDeleter});
        }
    }
    virtual nvinfer1::IPluginV2* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const char* libNamespace) override
    {
        assert(isPluginV2(layerName));
        PluginLayerType ltype = getPluginTypev2(layerName);
        assert(nbWeights == 0 && weights == nullptr);
        int i;
        if (ltype == PluginLayerType::kLEAKY)
        {
            i = mReLULayerNum1;
            assert(i >= 0 && i < NUM_RELU_IN_YOLO_V2);
            assert(mPluginLRelu[i] == nullptr);
            mPluginLRelu[i] = std::unique_ptr<IPluginV2, void (*)(IPluginV2*)>(createLReLUPlugin(kNEG_SLOPEV2), pluginDeleter);
            mReLULayerNum1++;
            mPluginLRelu[i].get()->setPluginNamespace(libNamespace);
            return mPluginLRelu[i].get();
        }
        else if (ltype == PluginLayerType::kREORG)
        {
            mPluginReorg = std::unique_ptr<IPluginV2, void (*)(IPluginV2*)>(createReorgPlugin(2), pluginDeleter);
            mPluginReorg.get()->setPluginNamespace(libNamespace);
            return mPluginReorg.get();
        }
        else if (ltype == PluginLayerType::kREGION)
        {
            RegionParameters params{gNumBoundBox, 4, gOutputClsSize, gSmTree};
            mPluginRegion = std::unique_ptr<IPluginV2, void (*)(IPluginV2*)>(createRegionPlugin(params), pluginDeleter);
            mPluginRegion.get()->setPluginNamespace(libNamespace);
            return mPluginRegion.get();
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    // Caffe parser plugin implementation
    bool isPluginV2(const char* name) override
    {
        return (PluginLayerType::kINVALID != getPluginTypev2(name));
    }

    void destroyPlugin()
    {
        for (unsigned i = 0; i < NUM_RELU_IN_YOLO_V2; i++)
        {
            mPluginLRelu[i].reset();
        }
        mPluginReorg.reset();
        mPluginRegion.reset();
    }

private:
    void (*pluginDeleter)(IPluginV2*){[](IPluginV2* ptr) { ptr->destroy(); }};
    std::unique_ptr<IPluginV2, void (*)(IPluginV2*)> mPluginReorg;
    std::unique_ptr<IPluginV2, void (*)(IPluginV2*)> mPluginRegion;
    std::vector<std::unique_ptr<IPluginV2, void (*)(IPluginV2*)>> mPluginLRelu;
    int mReLULayerNum1 = 0;
};
