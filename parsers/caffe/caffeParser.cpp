#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cassert>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

#define CHECK_NULL(ptr)                                                                 \
    if ((ptr) == nullptr)                                                               \
    {                                                                                   \
        std::cout << "Error: input " << #ptr << " is NULL in " << FN_NAME << std::endl; \
        return;                                                                         \
    }
#define CHECK_NULL_RET_NULL(ptr)                                                        \
    if ((ptr) == nullptr)                                                               \
    {                                                                                   \
        std::cout << "Error: input " << #ptr << " is NULL in " << FN_NAME << std::endl; \
        return nullptr;                                                                 \
    }
#define CHECK_NULL_RET_VAL(ptr, val)                                                    \
    if ((ptr) == nullptr)                                                               \
    {                                                                                   \
        std::cout << "Error: input " << #ptr << " is NULL in " << FN_NAME << std::endl; \
        return val;                                                                     \
    }

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "ditcaffe.pb.h"
#include "parserHelper.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

#include "half.h"

#define RETURN_AND_LOG_ERROR(ret, message) RETURN_AND_LOG_ERROR_IMPL(ret, message, "CaffeParser: ")

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace parserhelper;
namespace dc = ditcaffe;
typedef half_float::half float16;

class BlobNameToTensor : public IBlobNameToTensor
{
public:
    void add(const std::string& name, ITensor* tensor)
    {
        mMap[name] = tensor;
    }

    ITensor* find(const char* name) const override
    {
        auto p = mMap.find(name);
        if (p == mMap.end())
        {
            return nullptr;
        }
        return p->second;
    }

    ITensor*& operator[](const std::string& name)
    {
        return mMap[name];
    }

    void setTensorNames()
    {
        for (auto& p : mMap)
        {
            p.second->setName(p.first.c_str());
        }
    }

    ~BlobNameToTensor() override = default;

    bool isOK()
    {
        return !mError;
    }

private:
    std::map<std::string, ITensor*> mMap;
    bool mError{false};
};

void nvcaffeparser1::shutdownProtobufLibrary()
{
    google::protobuf::ShutdownProtobufLibrary();
}

// There are some challenges associated with importing caffe models. One is that
// a .caffemodel file just consists of layers and doesn't have the specs for its
// input and output blobs.
//
// So we need to read the deploy file to get the input

bool readBinaryProto(dc::NetParameter* net, const char* file, size_t bufSize)
{
    CHECK_NULL_RET_VAL(net, false)
    CHECK_NULL_RET_VAL(file, false)
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in | std::ios::binary);
    if (!stream)
    {
        RETURN_AND_LOG_ERROR(false, "Could not open file " + std::string(file));
    }

    IstreamInputStream rawInput(&stream);
    CodedInputStream codedInput(&rawInput);
    codedInput.SetTotalBytesLimit(int(bufSize), -1);

    bool ok = net->ParseFromCodedStream(&codedInput);
    stream.close();

    if (!ok)
    {
        RETURN_AND_LOG_ERROR(false, "Could not parse binary model file");
    }

    return ok;
}

bool readTextProto(dc::NetParameter* net, const char* file)
{
    CHECK_NULL_RET_VAL(net, false)
    CHECK_NULL_RET_VAL(file, false)
    using namespace google::protobuf::io;

    std::ifstream stream(file, std::ios::in);
    if (!stream)
    {
        RETURN_AND_LOG_ERROR(false, "Could not open file " + std::string(file));
    }

    IstreamInputStream input(&stream);
    bool ok = google::protobuf::TextFormat::Parse(&input, net);
    stream.close();
    return ok;
}

enum class WeightType
{
    // types for convolution, deconv, fully connected
    kGENERIC = 0, // typical weights for the layer: e.g. filter (for conv) or matrix weights (for innerproduct)
    kBIAS = 1,    // bias weights

    // These enums are for BVLCCaffe, which are incompatible with nvCaffe enums below.
    // See batch_norm_layer.cpp in BLVC source of Caffe
    kMEAN = 0,
    kVARIANCE = 1,
    kMOVING_AVERAGE = 2,

    // These enums are for nvCaffe, which are incompatible with BVLCCaffe enums above
    // See batch_norm_layer.cpp in NVidia fork of Caffe
    kNVMEAN = 0,
    kNVVARIANCE = 1,
    kNVSCALE = 3,
    kNVBIAS = 4
};

template <typename INPUT, typename OUTPUT>
void* convertInternal(void** ptr, int64_t count, bool* mOK)
{
    assert(ptr != nullptr);
    if (*ptr == nullptr)
    {
        return nullptr;
    }
    if (!count)
    {
        return nullptr;
    }
    auto* iPtr = static_cast<INPUT*>(*ptr);
    auto* oPtr = static_cast<OUTPUT*>(malloc(count * sizeof(OUTPUT)));
    for (int i = 0; i < count; ++i)
    {
        if (static_cast<OUTPUT>(iPtr[i]) > std::numeric_limits<OUTPUT>::max()
            || static_cast<OUTPUT>(iPtr[i]) < std::numeric_limits<OUTPUT>::lowest())
        {
            std::cout << "Error: Weight " << iPtr[i] << " is outside of [" << std::numeric_limits<OUTPUT>::max()
                      << ", " << std::numeric_limits<OUTPUT>::lowest() << "]." << std::endl;
            if (mOK)
            {
                (*mOK) = false;
            }
            break;
        }
        oPtr[i] = iPtr[i];
    }
    (*ptr) = oPtr;
    return oPtr;
}

class CaffeWeightFactory
{
public:
    CaffeWeightFactory(const dc::NetParameter& msg, DataType dataType, std::vector<void*>& tmpAllocs, bool isInitialized)
        : mMsg(msg)
        , mTmpAllocs(tmpAllocs)
        , mDataType(dataType)
        , mInitialized(isInitialized)
    {
        mRef = std::unique_ptr<dc::NetParameter>(new dc::NetParameter);
    }

    DataType getDataType() const
    {
        return mDataType;
    }

    size_t getDataTypeSize() const
    {
        switch (getDataType())
        {
        case DataType::kFLOAT:
        case DataType::kINT32:
            return 4;
        case DataType::kHALF:
            return 2;
        case DataType::kINT8:
            return 1;
        }
        return 0;
    }

    std::vector<void*>& getTmpAllocs()
    {
        return mTmpAllocs;
    }

    int getBlobsSize(const std::string& layerName)
    {
        for (int i = 0, n = mMsg.layer_size(); i < n; ++i)
        {
            if (mMsg.layer(i).name() == layerName)
            {
                return mMsg.layer(i).blobs_size();
            }
        }
        return 0;
    }

    const dc::BlobProto* getBlob(const std::string& layerName, int index)
    {
        if (mMsg.layer_size() > 0)
        {
            for (int i = 0, n = mMsg.layer_size(); i < n; i++)
            {
                if (mMsg.layer(i).name() == layerName && index < mMsg.layer(i).blobs_size())
                {
                    return &mMsg.layer(i).blobs(index);
                }
            }
        }
        else
        {
            for (int i = 0, n = mMsg.layers_size(); i < n; i++)
            {
                if (mMsg.layers(i).name() == layerName && index < mMsg.layers(i).blobs_size())
                {
                    return &mMsg.layers(i).blobs(index);
                }
            }
        }

        return nullptr;
    }

    std::vector<Weights> getAllWeights(const std::string& layerName)
    {
        std::vector<Weights> v;
        for (int i = 0;; i++)
        {
            auto b = getBlob(layerName, i);
            if (b == nullptr)
            {
                break;
            }
            auto weights = getWeights(*b, layerName);
            convert(weights, DataType::kFLOAT);
            v.push_back(weights);
        }
        return v;
    }

    virtual Weights operator()(const std::string& layerName, WeightType weightType)
    {
        const dc::BlobProto* blobMsg = getBlob(layerName, int(weightType));
        if (blobMsg == nullptr)
        {
            std::cout << "Weights for layer " << layerName << " doesn't exist" << std::endl;
            RETURN_AND_LOG_ERROR(getNullWeights(), "ERROR: Attempting to access NULL weights");
            assert(0);
        }
        return getWeights(*blobMsg, layerName);
    }

    void convert(Weights& weights, DataType targetType)
    {
        void* tmpAlloc{nullptr};
        if (weights.type == DataType::kFLOAT && targetType == DataType::kHALF)
        {
            tmpAlloc = convertInternal<float, float16>(const_cast<void**>(&weights.values), weights.count, &mOK);
            weights.type = targetType;
        }
        if (weights.type == DataType::kHALF && targetType == DataType::kFLOAT)
        {
            tmpAlloc = convertInternal<float16, float>(const_cast<void**>(&weights.values), weights.count, &mOK);
            weights.type = targetType;
        }
        if (tmpAlloc)
        {
            mTmpAllocs.push_back(tmpAlloc);
        }
    }

    void convert(Weights& weights)
    {
        convert(weights, getDataType());
    }

    bool isOK()
    {
        return mOK;
    }

    bool isInitialized()
    {
        return mInitialized;
    }

    Weights getNullWeights()
    {
        return Weights{mDataType, nullptr, 0};
    }

    Weights allocateWeights(int64_t elems, std::uniform_real_distribution<float> distribution = std::uniform_real_distribution<float>(-0.01f, 0.01F))
    {
        void* data = malloc(elems * getDataTypeSize());

        switch (getDataType())
        {
        case DataType::kFLOAT:
            for (int64_t i = 0; i < elems; ++i)
            {
                ((float*) data)[i] = distribution(generator);
            }
            break;
        case DataType::kHALF:
            for (int64_t i = 0; i < elems; ++i)
            {
                ((float16*) data)[i] = (float16)(distribution(generator));
            }
            break;
        default:
            break;
        }

        mTmpAllocs.push_back(data);
        return Weights{getDataType(), data, elems};
    }

    Weights allocateWeights(int64_t elems, std::normal_distribution<float> distribution)
    {
        void* data = malloc(elems * getDataTypeSize());

        switch (getDataType())
        {
        case DataType::kFLOAT:
            for (int64_t i = 0; i < elems; ++i)
            {
                ((float*) data)[i] = distribution(generator);
            }
            break;
        case DataType::kHALF:
            for (int64_t i = 0; i < elems; ++i)
            {
                ((float16*) data)[i] = (float16)(distribution(generator));
            }
            break;
        default:
            break;
        }

        mTmpAllocs.push_back(data);
        return Weights{getDataType(), data, elems};
    }

    static dc::Type getBlobProtoDataType(const dc::BlobProto& blobMsg)
    {
        if (blobMsg.has_raw_data())
        {
            assert(blobMsg.has_raw_data_type());
            return blobMsg.raw_data_type();
        }
        if (blobMsg.double_data_size() > 0)
        {
            return dc::DOUBLE;
        }
        return dc::FLOAT;
    }

    static size_t sizeOfCaffeType(dc::Type type)
    {
        if (type == dc::FLOAT)
        {
            return sizeof(float);
        }
        if (type == dc::FLOAT16)
        {
            return sizeof(uint16_t);
        }
        return sizeof(double);
    }

    // The size returned here is the number of array entries, not bytes
    static std::pair<const void*, size_t> getBlobProtoData(const dc::BlobProto& blobMsg,
                                                           dc::Type type, std::vector<void*>& tmpAllocs)
    {
        // NVCaffe new binary format. It may carry any type.
        if (blobMsg.has_raw_data())
        {
            assert(blobMsg.has_raw_data_type());
            if (blobMsg.raw_data_type() == type)
            {
                return std::make_pair(&blobMsg.raw_data().front(),
                                      blobMsg.raw_data().size() / sizeOfCaffeType(type));
            }
        }
        // Old BVLC format.
        if (blobMsg.data_size() > 0 && type == dc::FLOAT)
        {
            return std::make_pair(&blobMsg.data().Get(0), blobMsg.data_size());
        }

        // Converting to the target type otherwise
        const int count = blobMsg.has_raw_data() ? blobMsg.raw_data().size() / sizeOfCaffeType(blobMsg.raw_data_type()) : (blobMsg.data_size() > 0 ? blobMsg.data_size() : blobMsg.double_data_size());

        if (count > 0)
        {
            void* new_memory = malloc(count * sizeOfCaffeType(type));
            tmpAllocs.push_back(new_memory);

            if (type == dc::FLOAT)
            {
                auto* dst = reinterpret_cast<float*>(new_memory);
                if (blobMsg.has_raw_data())
                {
                    if (blobMsg.raw_data_type() == dc::FLOAT16)
                    {
                        const auto* src = reinterpret_cast<const float16*>(&blobMsg.raw_data().front());
                        for (int i = 0; i < count; ++i)
                        {
                            dst[i] = float(src[i]);
                        }
                    }
                    else if (blobMsg.raw_data_type() == dc::DOUBLE)
                    {
                        const auto* src = reinterpret_cast<const double*>(&blobMsg.raw_data().front());
                        for (int i = 0; i < count; ++i)
                        {
                            dst[i] = float(src[i]);
                        }
                    }
                }
                else if (blobMsg.double_data_size() == count)
                {
                    for (int i = 0; i < count; ++i)
                    {
                        dst[i] = float(blobMsg.double_data(i));
                    }
                }
                return std::make_pair(new_memory, count);
            }
            if (type == dc::FLOAT16)
            {
                auto* dst = reinterpret_cast<float16*>(new_memory);

                if (blobMsg.has_raw_data())
                {
                    if (blobMsg.raw_data_type() == dc::FLOAT)
                    {
                        const auto* src = reinterpret_cast<const float*>(&blobMsg.raw_data().front());
                        for (int i = 0; i < count; ++i)
                        {
                            dst[i] = float16(src[i]);
                        }
                    }
                    else if (blobMsg.raw_data_type() == dc::DOUBLE)
                    {
                        const auto* src = reinterpret_cast<const double*>(&blobMsg.raw_data().front());
                        for (int i = 0; i < count; ++i)
                        {
                            dst[i] = float16(float(src[i]));
                        }
                    }
                }
                else if (blobMsg.data_size() == count)
                {
                    for (int i = 0; i < count; ++i)
                    {
                        dst[i] = float16(blobMsg.data(i));
                    }
                }
                else if (blobMsg.double_data_size() == count)
                {
                    for (int i = 0; i < count; ++i)
                    {
                        dst[i] = float16(float(blobMsg.double_data(i)));
                    }
                }
                return std::make_pair(new_memory, count);
            }
        }
        return std::make_pair(nullptr, 0UL);
    }

private:
    template <typename T>
    bool checkForNans(const void* values, int count, const std::string& layerName)
    {
        const T* v = reinterpret_cast<const T*>(values);
        for (int i = 0; i < count; i++)
        {
            if (std::isnan(float(v[i])))
            {
                std::cout << layerName << ": Nan detected in weights" << std::endl;
                return false;
            }
        }
        return true;
    }

    Weights getWeights(const dc::BlobProto& blobMsg, const std::string& layerName)
    {
        // Always load weights into FLOAT format
        const auto blobProtoData = getBlobProtoData(blobMsg, dc::FLOAT, mTmpAllocs);

        if (blobProtoData.first == nullptr)
        {
            const int bits = mDataType == DataType::kFLOAT ? 32 : 16;
            std::cout << layerName << ": ERROR - " << bits << "-bit weights not found for "
                      << bits << "-bit model" << std::endl;
            mOK = false;
            return Weights{DataType::kFLOAT, nullptr, 0};
        }

        mOK &= checkForNans<float>(blobProtoData.first, int(blobProtoData.second), layerName);
        return Weights{DataType::kFLOAT, blobProtoData.first, int(blobProtoData.second)};
    }

    const dc::NetParameter& mMsg;
    std::unique_ptr<dc::NetParameter> mRef;
    std::vector<void*>& mTmpAllocs;
    DataType mDataType;
    // bool mQuantize;
    bool mInitialized;
    std::default_random_engine generator;

    bool mOK{true};
};

bool checkBlobs(const dc::LayerParameter& msg, int bottoms, int tops)
{
    if (msg.bottom_size() != bottoms)
    {
        std::cout << msg.name() << ": expected " << bottoms << " bottom blobs, found " << msg.bottom_size() << std::endl;
        return false;
    }

    if (msg.top_size() != tops)
    {
        std::cout << msg.name() << ": expected " << tops << " tops blobs, found " << msg.top_size() << std::endl;
        return false;
    }
    return true;
}

ILayer* parseConvolution(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::ConvolutionParameter& p = msg.convolution_param();
    int nbOutputs = p.num_output();

    int kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size(0);
    int kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size_size() > 1 ? p.kernel_size(1) : p.kernel_size(0);
    int C = getCHW(tensors[msg.bottom(0)]->getDimensions()).c();
    int G = p.has_group() ? p.group() : 1;

    auto CbyG = float(C / G * nbOutputs);
    float std_dev = 1.0F / sqrtf((kernelW * kernelH * sqrtf(CbyG)));
    Weights kernelWeights = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) : weightFactory.allocateWeights(kernelW * kernelH * CbyG, std::normal_distribution<float>(0.0F, std_dev));
    Weights biasWeights = !p.has_bias_term() || p.bias_term() ? (weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kBIAS) : weightFactory.allocateWeights(nbOutputs)) : weightFactory.getNullWeights();

    weightFactory.convert(kernelWeights);
    weightFactory.convert(biasWeights);
    auto layer = network.addConvolution(*tensors[msg.bottom(0)], nbOutputs, DimsHW{kernelH, kernelW}, kernelWeights, biasWeights);

    if (layer)
    {
        int strideH = p.has_stride_h() ? p.stride_h() : p.stride_size() > 0 ? p.stride(0) : 1;
        int strideW = p.has_stride_w() ? p.stride_w() : p.stride_size() > 1 ? p.stride(1) : p.stride_size() > 0 ? p.stride(0) : 1;

        int padH = p.has_pad_h() ? p.pad_h() : p.pad_size() > 0 ? p.pad(0) : 0;
        int padW = p.has_pad_w() ? p.pad_w() : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;

        int dilationH = p.dilation_size() > 0 ? p.dilation(0) : 1;
        int dilationW = p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;

        layer->setStride(DimsHW{strideH, strideW});
        layer->setPadding(DimsHW{padH, padW});
		layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN);
        layer->setDilation(DimsHW{dilationH, dilationW});

        layer->setNbGroups(G);
    }
    return layer;
}

ILayer* parsePooling(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::PoolingParameter& p = msg.pooling_param();
    if (p.pool() != dc::PoolingParameter::MAX && p.pool() != dc::PoolingParameter::AVE)
    {
        std::cout << "Caffe Parser: only AVE and MAX pool operations are supported" << std::endl;
        return nullptr;
    }

    int kernelH, kernelW;
    if (p.has_global_pooling() && p.global_pooling())
    {
        DimsCHW dims = getCHW(tensors[msg.bottom(0)]->getDimensions());
        kernelH = dims.h();
        kernelW = dims.w();
    }
    else
    {
        // mandatory
        kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size();
        kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size();
    }

    PoolingType type = p.has_pool() && p.pool() == dc::PoolingParameter::AVE ? PoolingType::kAVERAGE : PoolingType::kMAX;
    auto layer = network.addPooling(*tensors[msg.bottom(0)], type, DimsHW{kernelH, kernelW});

    if (layer)
    {
        int stride = p.has_stride() ? p.stride() : 1;
        layer->setStride(DimsHW{p.has_stride_h() ? int(p.stride_h()) : stride, p.has_stride_w() ? int(p.stride_w()) : stride});

        int pad = p.has_pad() ? p.pad() : 0;
        layer->setPadding(DimsHW{p.has_pad_h() ? int(p.pad_h()) : pad, p.has_pad_w() ? int(p.pad_w()) : pad});

        layer->setName(msg.name().c_str());
		layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_UP); // caffe pool use ceil mode by default
        // FB pooling parameters
        // Use floor((height + 2 * padding - kernel) / stride) + 1
        // instead of ceil((height + 2 * padding - kernel) / stride) + 1
        if (p.has_torch_pooling() ? p.torch_pooling() : false)
        {
		    layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN); // facebook torch pool use floor mode
		}

        tensors[msg.top(0)] = layer->getOutput(0);

        layer->setAverageCountExcludesPadding(false); // unlike other frameworks, caffe use inclusive counting for padded averaging
    }
    return layer;
}

ILayer* parseInnerProduct(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    const dc::InnerProductParameter& p = msg.inner_product_param();

    int64_t nbInputs = volume(getCHW(tensors[msg.bottom(0)]->getDimensions()));
    int64_t nbOutputs = p.num_output();

    float std_dev = 1.0F / sqrtf(nbInputs * nbOutputs);
    Weights kernelWeights = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) : weightFactory.allocateWeights(nbInputs * nbOutputs, std::normal_distribution<float>(0.0F, std_dev));
    Weights biasWeights = !p.has_bias_term() || p.bias_term() ? (weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kBIAS) : weightFactory.allocateWeights(nbOutputs)) : weightFactory.getNullWeights();

    weightFactory.convert(kernelWeights);
    weightFactory.convert(biasWeights);
    return network.addFullyConnected(*tensors[msg.bottom(0)], p.num_output(), kernelWeights, biasWeights);
}

ILayer* parseReLU(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::ReLUParameter& p = msg.relu_param();

    if (p.has_negative_slope() && p.negative_slope() != 0)
    {
        auto newLayer = network.addActivation(*tensors[msg.bottom(0)], ActivationType::kLEAKY_RELU);
        newLayer->setAlpha(p.negative_slope());
        return newLayer;
    }
    return network.addActivation(*tensors[msg.bottom(0)], ActivationType::kRELU);
}

ILayer* parseELU(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::ELUParameter& p = msg.elu_param();

    float alpha = 1.f; // default parameter
    if (p.has_alpha())
    {
        alpha = p.alpha();
    }
    auto newLayer = network.addActivation(*tensors[msg.bottom(0)], ActivationType::kELU);
    newLayer->setAlpha(alpha);
    return newLayer;
}

ILayer* parseAbsVal(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }
    return network.addUnary(*tensors[msg.bottom(0)], UnaryOperation::kABS);
}

ILayer* parsePReLU(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory ,
                   BlobNameToTensor& tensors)
{
    // Caffe stores the slopes as weights rather than as a tensor, and only supports different slopes
    // per channel
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::PReLUParameter& p = msg.prelu_param();
    bool channelShared = p.has_channel_shared() ? p.channel_shared() : false;
    auto inputDims = tensors[msg.bottom(0)]->getDimensions();
    if (inputDims.nbDims < 2)
    {
        return nullptr;
    }
    int nWeights = channelShared ? 1 : inputDims.d[1]; // Caffe treats second input dimension as channels
    Dims slopesDims{inputDims.nbDims, {1}, {DimensionType::kSPATIAL}};
    slopesDims.d[1] = nWeights;

    Weights w = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) :
                weightFactory.allocateWeights(nWeights, std::uniform_real_distribution<float>(0.F, 1.F));
    auto constLayer = network.addConstant(slopesDims, w);
    return network.addParametricReLU(*tensors[msg.bottom(0)], *constLayer->getOutput(0));
}

ILayer* parseClip(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }
    const dc::ClipParameter& p = msg.clip_param();
    float alpha = std::numeric_limits<float>::lowest(); // lower bound
    float beta = std::numeric_limits<float>::max();     // upper bound
    if(p.has_min())
    {
        alpha = p.min();
    }
    if(p.has_max())
    {
        beta = p.max();
    }
    auto layer = network.addActivation(*tensors[msg.bottom(0)], ActivationType::kCLIP);
    layer->setAlpha(alpha);
    layer->setBeta(beta);
    return layer;
}

ILayer* parseBNLL(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /* weightFactory */, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }
    return network.addActivation(*tensors[msg.bottom(0)], ActivationType::kSOFTPLUS);
}

ILayer* parseSoftMax(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::SoftmaxParameter& p = msg.softmax_param();

    // Caffe supports negative axis, indexing from the last dimension
    // However, there is a discrepancy in the internal tensor dimension in some cases.
    // For example. InnerProduct produces flat 1D blob in Caffe, while TensorRT still
    // produces CHW format. MNIST sample generates input to Softmax as,
    //     Caffe    = n x 10
    //     TensorRT = n x 10 x 1 x 1
    // To make sure we do not run into issues, negative axis won't be supported in TensorRT
    int nbDims = tensors[msg.bottom(0)]->getDimensions().nbDims;
    bool hasAxis = p.has_axis();       // optional parameter
    int axis = hasAxis ? p.axis() : 1; // default is 1

    bool axisAbort = (axis <= 0) || (axis > 3) || (axis > nbDims);

    if (axisAbort)
    {
        std::cout << "Caffe Parser: Invalid axis in softmax layer - Cannot perform softmax along batch size dimension and expects NCHW input. Negative axis is not supported in TensorRT, please use positive axis indexing" << std::endl;
        return nullptr;
    }

    auto softmax = network.addSoftMax(*tensors[msg.bottom(0)]);
    // Do this so that setAxes is not used when the default axis is needed
    // This is necessary to preserve correct roll-into-the-batch dimension behaviour for samples like FasterRCNN
    // NCHW -> default axis when setAxes is not called will be 1 (the C dimension)
    // NPCHW -> default axis when setAxes is not called will be 2 (the C dimension)
    if (hasAxis)
    {
        uint32_t axes = 1u << (axis - 1);
        softmax->setAxes(axes);
    }
    return softmax;
}

ILayer* parseLRN(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::LRNParameter& p = msg.lrn_param();
    int localSize = p.has_local_size() ? p.local_size() : 5;
    float alpha = p.has_alpha() ? p.alpha() : 1;
    float beta = p.has_beta() ? p.beta() : 5;
    float k = p.has_k() ? p.k() : 1;

    return network.addLRN(*tensors[msg.bottom(0)], localSize, alpha, beta, k);
}

ILayer* parsePower(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::PowerParameter& p = msg.power_param();

    float shift = p.has_shift() ? p.shift() : 0.0f;
    float scale = p.has_scale() ? p.scale() : 1.0f;
    float power = p.has_power() ? p.power() : 1.0f;

    DataType dataType = weightFactory.getDataType();
    assert(dataType == DataType::kFLOAT || dataType == DataType::kHALF);

    Weights wShift, wScale, wPower;
    if (dataType == DataType::kHALF)
    {
        auto* t = reinterpret_cast<float16*>(malloc(3 * sizeof(float16)));
        t[0] = float16(shift), t[1] = float16(scale), t[2] = float16(power);
        wShift = Weights{DataType::kHALF, &t[0], 1};
        wScale = Weights{DataType::kHALF, &t[1], 1};
        wPower = Weights{DataType::kHALF, &t[2], 1};
        weightFactory.getTmpAllocs().push_back(t);
    }
    else
    {
        auto* t = reinterpret_cast<float*>(malloc(3 * sizeof(float)));
        t[0] = shift, t[1] = scale, t[2] = power;
        wShift = Weights{DataType::kFLOAT, &t[0], 1};
        wScale = Weights{DataType::kFLOAT, &t[1], 1};
        wPower = Weights{DataType::kFLOAT, &t[2], 1};
        weightFactory.getTmpAllocs().push_back(t);
    }

    weightFactory.convert(wShift);
    weightFactory.convert(wScale);
    weightFactory.convert(wPower);
    return network.addScale(*tensors[msg.bottom(0)], ScaleMode::kUNIFORM, wShift, wScale, wPower);
}

ILayer* parseEltwise(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 2, 1))
    {
        return nullptr;
    }

    const dc::EltwiseParameter& p = msg.eltwise_param();

    ElementWiseOperation op = ElementWiseOperation::kSUM;
    switch (p.operation())
    {
    case dc::EltwiseParameter_EltwiseOp_SUM: op = ElementWiseOperation::kSUM; break;
    case dc::EltwiseParameter_EltwiseOp_PROD: op = ElementWiseOperation::kPROD; break;
    case dc::EltwiseParameter_EltwiseOp_MAX: op = ElementWiseOperation::kMAX; break;
    }

    return network.addElementWise(*tensors[msg.bottom(0)], *tensors[msg.bottom(1)], op);
}

ILayer* parseConcat(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    const dc::ConcatParameter& p = msg.concat_param();
    bool hasAxis = p.has_axis(); // optional parameter

    if (hasAxis && p.axis() <= 0)
    {
        std::cout << "Caffe parser: Concat along batch axis or negative axis is not supported." << std::endl;
        return nullptr;
    }

    std::vector<ITensor*> ptrs;
    for (unsigned int i = 0, n = msg.bottom_size(); i < n; i++)
    {
        ptrs.push_back(tensors[msg.bottom().Get(i)]);
    }

    auto concat = network.addConcatenation(&ptrs[0], msg.bottom_size());

    // If no axis is explicitly provided, do not call setAxis.
    // Rely on the default axis setting inside TRT which takes into account NPCHW and higher dimensional input.
    if (hasAxis)
    {
        concat->setAxis(p.axis() - 1);
    }

    return concat;
}

ILayer* parseDeconvolution(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::ConvolutionParameter& p = msg.convolution_param();
    int nbOutputs = p.num_output();
    int nbGroups = p.has_group() ? p.group() : 1;

    int dilationH = p.dilation_size() > 0 ? p.dilation(0) : 1;
    int dilationW = p.dilation_size() > 1 ? p.dilation(1) : p.dilation_size() > 0 ? p.dilation(0) : 1;
    if (dilationH != 1 || dilationW != 1)
    {
        RETURN_AND_LOG_ERROR(nullptr, "Dilated deconvolution is not supported.");
    }

    int kernelW = p.has_kernel_w() ? p.kernel_w() : p.kernel_size(0);
    int kernelH = p.has_kernel_h() ? p.kernel_h() : p.kernel_size_size() > 1 ? p.kernel_size(1) : p.kernel_size(0);
    int C = getCHW(tensors[msg.bottom(0)]->getDimensions()).c();

    float std_dev = 1.0F / sqrtf(kernelW * kernelH * sqrtf(C * nbOutputs));
    Weights kernelWeights = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) : weightFactory.allocateWeights(kernelW * kernelH * C * nbOutputs / nbGroups, std::normal_distribution<float>(0.0F, std_dev));
    Weights biasWeights = !p.has_bias_term() || p.bias_term() ? (weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kBIAS) : weightFactory.allocateWeights(nbOutputs)) : weightFactory.getNullWeights();

    weightFactory.convert(kernelWeights);
    weightFactory.convert(biasWeights);
    auto layer = network.addDeconvolution(*tensors[msg.bottom(0)], nbOutputs, DimsHW{kernelH, kernelW}, kernelWeights, biasWeights);

    if (layer)
    {
        int strideW = p.has_stride_w() ? p.stride_w() : p.stride_size() > 0 ? p.stride(0) : 1;
        int strideH = p.has_stride_h() ? p.stride_h() : p.stride_size() > 1 ? p.stride(1) : p.stride_size() > 0 ? p.stride(0) : 1;

        int padW = p.has_pad_w() ? p.pad_w() : p.pad_size() > 0 ? p.pad(0) : 0;
        int padH = p.has_pad_h() ? p.pad_h() : p.pad_size() > 1 ? p.pad(1) : p.pad_size() > 0 ? p.pad(0) : 0;

        layer->setStride(DimsHW{strideH, strideW});
        layer->setPadding(DimsHW{padH, padW});
		layer->setPaddingMode(PaddingMode::kCAFFE_ROUND_DOWN);
        layer->setNbGroups(nbGroups);

        layer->setKernelWeights(kernelWeights);
        if (!p.has_bias_term() || p.bias_term())
        {
            layer->setBiasWeights(biasWeights);
        }
    }
    return layer;
}

ILayer* parseSigmoid(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    return network.addActivation(*tensors[msg.bottom(0)], ActivationType::kSIGMOID);
}

ILayer* parseTanH(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    return network.addActivation(*tensors[msg.bottom(0)], ActivationType::kTANH);
}

template <typename T>
inline bool bnConvertWrap(float scaleFactor, const Weights& variance, const Weights& mean,
                          const Weights& scaleBlob, const Weights& biasBlob,
                          Weights& shift, Weights& scale, float eps,
                          bool nvCaffe, CaffeWeightFactory& weightFactory)
{

    assert(shift.count == scale.count);
    if (nvCaffe)
    {
        if (scaleBlob.values == nullptr)
        {
            return false;
        }
        if (biasBlob.values == nullptr)
        {
            return false;
        }
    }
    T* shiftv = reinterpret_cast<T*>(malloc(sizeof(T) * shift.count));
    if (!shiftv)
    {
        return false;
    }

    T* scalev = reinterpret_cast<T*>(malloc(sizeof(T) * scale.count));
    if (!scalev)
    {
        free(shiftv);
        return false;
    }
    shift.values = shiftv;
    scale.values = scalev;
    weightFactory.getTmpAllocs().push_back(shiftv);
    weightFactory.getTmpAllocs().push_back(scalev);

    const T* m = reinterpret_cast<const T*>(mean.values);
    const T* v = reinterpret_cast<const T*>(variance.values);
    for (int i = 0; i < shift.count; i++)
    {
        scalev[i] = T(1.0f / std::sqrt(float(v[i]) * scaleFactor + eps));
        shiftv[i] = T(-(float(m[i]) * scaleFactor * float(scalev[i])));
    }

    if (nvCaffe)
    {
        const T* s = reinterpret_cast<const T*>(scaleBlob.values);
        const T* b = reinterpret_cast<const T*>(biasBlob.values);
        for (int i = 0; i < shift.count; i++)
        {
            scalev[i] = T(float(scalev[i]) * s[i]);
            shiftv[i] = T(float(shiftv[i]) * s[i]) + b[i];
        }
    }
    return true;
}

ILayer* parseBatchNormalization(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::BatchNormParameter& p = msg.batch_norm_param();
    bool nvCaffe = weightFactory.getBlobsSize(msg.name()) == 5;

    int C = getCHW(tensors[msg.bottom(0)]->getDimensions()).c();

    Weights mean{DataType::kFLOAT, nullptr, 0},
        variance{DataType::kFLOAT, nullptr, 0},
        scaleBlob{DataType::kFLOAT, nullptr, 0},
        biasBlob{DataType::kFLOAT, nullptr, 0},
        movingAverage{DataType::kFLOAT, nullptr, 0};

    // Because of the incompatible nature of the batch normalizations
    // between BLVC Caffe and nvCaffe, two different paths have to be
    // used.
    if (nvCaffe)
    {
        if (weightFactory.isInitialized())
        {
            mean = weightFactory(msg.name(), WeightType::kNVMEAN);
            variance = weightFactory(msg.name(), WeightType::kNVVARIANCE);
            scaleBlob = weightFactory(msg.name(), WeightType::kNVSCALE);
            biasBlob = weightFactory(msg.name(), WeightType::kNVBIAS);
        }
        else
        {
            mean = weightFactory.allocateWeights(C);
            variance = weightFactory.allocateWeights(C, std::uniform_real_distribution<float>(0.9F, 1.1F));
            scaleBlob = weightFactory.allocateWeights(C, std::uniform_real_distribution<float>(0.9F, 1.1F));
            biasBlob = weightFactory.allocateWeights(C);
        }
    }
    else
    {
        if (weightFactory.isInitialized())
        {
            mean = weightFactory(msg.name(), WeightType::kMEAN);
            variance = weightFactory(msg.name(), WeightType::kVARIANCE);
            movingAverage = weightFactory(msg.name(), WeightType::kMOVING_AVERAGE);
        }
        else
        {
            mean = weightFactory.allocateWeights(C);
            variance = weightFactory.allocateWeights(C, std::uniform_real_distribution<float>(0.9F, 1.1F));
            movingAverage = weightFactory.allocateWeights(1, std::uniform_real_distribution<float>(0.99F, 1.01F));
        }
        assert(mean.count == variance.count && movingAverage.count == 1);
    }

    Weights shift{mean.type, nullptr, mean.count};
    Weights scale{mean.type, nullptr, mean.count};
    Weights power{mean.type, nullptr, 0};
    bool success{false};
    float scaleFactor{1.0f};
    if (!nvCaffe)
    {
        float average{0.0f};
        // Inside weightFactory, the weights are generated based off the type.
        if (mean.type == DataType::kFLOAT)
        {
            average = *(static_cast<const float*>(movingAverage.values));
        }
        else
        {
            average = *(static_cast<const float16*>(movingAverage.values));
        }
        if (average == 0.0f)
        {
            std::cout << "Batch normalization moving average is zero" << std::endl;
            return nullptr;
        }
        scaleFactor /= average;
    }
    if (mean.type == DataType::kFLOAT)
    {
        success = bnConvertWrap<float>(scaleFactor, variance, mean, scaleBlob, biasBlob, shift, scale, p.eps(), nvCaffe, weightFactory);
    }
    else
    {
        success = bnConvertWrap<float16>(scaleFactor, variance, mean, scaleBlob, biasBlob, shift, scale, p.eps(), nvCaffe, weightFactory);
    }

    if (!success)
    {
        return nullptr;
    }

    weightFactory.convert(shift);
    weightFactory.convert(scale);
    weightFactory.convert(power);
    return network.addScale(*tensors[msg.bottom(0)], ScaleMode::kCHANNEL, shift, scale, power);
}

ILayer* parseScale(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::ScaleParameter& p = msg.scale_param();
    int C = getCHW(tensors[msg.bottom(0)]->getDimensions()).c();

    Weights scale = weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kGENERIC) : weightFactory.allocateWeights(C, std::uniform_real_distribution<float>(0.9F, 1.1F));
    Weights shift = !p.has_bias_term() || p.bias_term() ? (weightFactory.isInitialized() ? weightFactory(msg.name(), WeightType::kBIAS) : weightFactory.allocateWeights(C)) : weightFactory.getNullWeights();
    Weights power = weightFactory.getNullWeights();
    weightFactory.convert(shift);
    weightFactory.convert(scale);
    weightFactory.convert(power);
    return network.addScale(*tensors[msg.bottom(0)], ScaleMode::kCHANNEL, shift, scale, power);
}

ILayer* parseCrop(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    // To crop, elements of the first bottom are selected to fit the dimensions
    // of the second, reference bottom. The crop is configured by
    // - the crop `axis` to pick the dimensions for cropping
    // - the crop `offset` to set the shift for all/each dimension
    // to align the cropped bottom with the reference bottom.
    // All dimensions up to but excluding `axis` are preserved, while
    // the dimensions including and trailing `axis` are cropped.
    // If only one `offset` is set, then all dimensions are offset by this amount.
    // Otherwise, the number of offsets must equal the number of cropped axes to
    // shift the crop in each dimension accordingly.
    // Note: standard dimensions are N,C,H,W so the default is a spatial crop,
    // and `axis` may be negative to index from the end (e.g., -1 for the last
    // axis).

    if (!checkBlobs(msg, 2, 1))
    {
        return nullptr;
    }

    // ONLY IMPLEMENT SPATIAL CROPPING
    // IF CROP LAYER IS NOT SPATIAL CROP, ABORT
    const dc::CropParameter& p = msg.crop_param();
    DimsCHW inputDims = getCHW(tensors[msg.bottom(0)]->getDimensions());
    DimsCHW refDims = getCHW(tensors[msg.bottom(1)]->getDimensions());
    bool hasAxis = p.has_axis();         // optional parameter
    int axis = hasAxis ? p.axis() : 2;   // default is 2 - spatial crop
    axis = (axis < 0) ? 4 + axis : axis; // axis negative number correction

    // acceptable axis values: 2, 3, -1, -2
    // unacceptable axis values: 0, 1, -3, -4 and anything else
    // acceptable corrected axis values: 2, 3
    // unacceptable corrected axis values: 0, 1 and anything else
    // protect against "garbage" input arguments
    bool axis_abort = (axis != 2 && axis != 3);

    // must be at least one offset
    // if only one offset, the same offset applies to all the dimensions
    // including the chosen axis and trailing it
    // if more than one offset, the number of offsets must match the number
    // of dimensions consisting of the axis and all the dimensions trailing it
    int num_offsets = p.offset_size();

    // 1 + (3 - axis) = 4 - axis
    // this is only valid for acceptable corrected axis values
    // if !axis_abort then invariant that num_dims == 1 || num_dims == 2
    int num_dims = 4 - axis;
    bool offset_abort = (num_offsets != 0 && num_offsets != 1 && num_offsets != num_dims);

    if (axis_abort)
    {
        std::cout << "Caffe Parser: Invalid axis in crop layer - only spatial cropping is supported" << std::endl;
        return nullptr;
    }

    if (offset_abort)
    {
        std::cout << "Caffe Parser: Invalid number of offsets in crop layer" << std::endl;
        return nullptr;
    }

    // get the offsets
    // the offsets are zero by default (in case no offset is specified)
    int offsetHeight = 0;
    int offsetWidth = 0;

    if (num_offsets != 0)
    {
        // offsetHeight will only be specified if the H channel is the chosen axis
        // in this case, regardless of whether there are one or multiple offsets
        // offsetHeight should always be the zero-indexed offset
        offsetHeight = axis == 2 ? p.offset(0) : 0;
        // offsetWidth should always be specified
        // if there is only one offset, use the zero-indexed offset
        // otherwise, use the one-indexed offset since the zero-indexed offet
        // is for offsetHeight
        offsetWidth = num_offsets == 1 ? p.offset(0) : p.offset(1);
    }

    // now compute the prePadding and postPadding required to perform the crop
    // so that the first bottom is the same spatial size as the second bottom
    // prePadding is the padding to the left/bottom (assuming origin is lower-left).
    // postPadding is the padding to the right/top.
    // - ( inputDims.h() - refDims.h() - offsetHeight ) = -inputDims.h() + refDims.h() + offsetHeight
    // - ( inputDims.w() - refDims.w() - offsetWidth ) = -inputDims.w() + refDims.w() + offsetWidth
    int prePadHeight = -offsetHeight;
    int prePadWidth = -offsetWidth;
    int postPadHeight = -inputDims.h() + refDims.h() + offsetHeight;
    int postPadWidth = -inputDims.w() + refDims.w() + offsetWidth;

    DimsHW prePadding = DimsHW{prePadHeight, prePadWidth};
    DimsHW postPadding = DimsHW{postPadHeight, postPadWidth};
    return network.addPadding(*tensors[msg.bottom(0)], prePadding, postPadding);
}

ILayer* parseReduction(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    // The first axis to reduce to a scalar -- may be negative to index from the
    // end (e.g., -1 for the last axis).
    // (Currently, only reduction along ALL "tail" axes is supported; reduction
    // of axis M through N, where N < num_axes - 1, is unsupported.)
    // Suppose we have an n-axis bottom Blob with shape:
    //     (d0, d1, d2, ..., d(m-1), dm, d(m+1), ..., d(n-1)).
    // If axis == m, the output Blob will have shape
    //     (d0, d1, d2, ..., d(m-1)),
    // and the ReductionOp operation is performed (d0 * d1 * d2 * ... * d(m-1))
    // times, each including (dm * d(m+1) * ... * d(n-1)) individual data.
    // If axis == 0 (the default), the output Blob always has the empty shape
    // (count 1), performing reduction across the entire input --
    // often useful for creating new loss functions.
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    // operation == 1 is SUM -> ReduceOperation::kSUM
    const int SUM = 1;
    // operation == 2 is ASUM -> UnaryOperation::kABS and ReduceOperation::kSUM
    const int ASUM = 2;
    // operation == 3 is SUMSQ -> ElementWiseOperation::kPROD and ReduceOperation::kSUM
    const int SUMSQ = 3;
    // operation == 4 is MEAN -> ReduceOperation::kAVG
    const int MEAN = 4;

    const dc::ReductionParameter& p = msg.reduction_param();
    bool hasOperation = p.has_operation();              // optional parameter
    bool hasAxis = p.has_axis();                        // optional parameter
    bool hasCoeff = p.has_coeff();                      // optional parameter
    int operation = hasOperation ? p.operation() : SUM; // default is SUM
    int axis = hasAxis ? p.axis() : 0;                  // default is 0
    axis = (axis < 0) ? 4 + axis : axis;                // axis negative number correction
    float coeff = hasCoeff ? p.coeff() : 1.0;           // default is 1

    // acceptable axis values: 1, 2, 3, -1, -2, -3
    // unacceptable axis values: 0 and anything else
    // acceptable corrected axis values: 1, 2, 3
    // unacceptable corrected axis values: 0 and anything else
    // protect against "garbage" input arguments
    bool axisAbort = (axis != 1 && axis != 2 && axis != 3);

    if (axisAbort)
    {
        std::cout << "Caffe Parser: Invalid axis in reduction layer - cannot reduce over batch size dimension and can only reduce NCHW input" << std::endl;
        return nullptr;
    }

    ReduceOperation op = (operation == MEAN ? ReduceOperation::kAVG : ReduceOperation::kSUM);
    // corrected axis values are 1, 2, 3
    // only reduction along tail dimensions is supported
    // 1 means 111 or 4 + 2 + 1 = 7
    // 2 means 110 or 4 + 2 = 6
    // 3 means 100 or 4
    // Let's employ a bit shift trick instead
    // 1000 = 8
    // axis == 1: 1u << (axis - 1) is 1 and so 8 - 1 = 7 or 111
    // axis == 2: 1u << (axis - 1) is 2 and so 8 - 2 = 6 or 110
    // axis == 3: 1u << (axis - 1) is 4 and so 8 - 4 = 4 or 100
    uint32_t reduceAxes = 8 - (1u << (axis - 1));

    ITensor* input = tensors[msg.bottom(0)];
    ILayer* returnVal = nullptr;
    // need to add in layer before for ASUM and SUMSQ
    if (operation == ASUM)
    {
        returnVal = network.addUnary(*input, UnaryOperation::kABS);
        input = returnVal->getOutput(0);
        std::string layerName = msg.name() + std::string("/reductionLayer/unaryLayer");
        returnVal->setName(layerName.c_str());
    }
    else if (operation == SUMSQ)
    {
        returnVal = network.addElementWise(*input, *input, ElementWiseOperation::kPROD);
        input = returnVal->getOutput(0);
        std::string layerName = msg.name() + std::string("/reductionLayer/elementWiseLayer");
        returnVal->setName(layerName.c_str());
    }

// add in the actual reduce layer
#define GIE_3111 0
#if GIE_3111
    returnVal = network.addReduce(*input, op, reduceAxes, false);
#else
    returnVal = network.addReduce(*input, op, reduceAxes, true);
    // output a warning
    std::cout << "Warning: The Reduce layer does not discard reduced dimensions. The reduced dimensions are treated as dimensions of size one in the output of the Reduce layer." << std::endl;
#endif
    input = returnVal->getOutput(0);
    std::string reduceLayerName = msg.name() + std::string("/reductionLayer/reduceLayer");
    returnVal->setName(reduceLayerName.c_str());

    // need to add in layer after for coeff != 1.0
    if (coeff != 1.0f)
    {
        auto* shiftArr = (float*) malloc(sizeof(float));
        auto* scaleArr = (float*) malloc(sizeof(float));
        auto* powerArr = (float*) malloc(sizeof(float));
        weightFactory.getTmpAllocs().push_back(shiftArr);
        weightFactory.getTmpAllocs().push_back(scaleArr);
        weightFactory.getTmpAllocs().push_back(powerArr);
        *shiftArr = 0.0f;
        *scaleArr = coeff;
        *powerArr = 1.0f;

        Weights wShift, wScale, wPower;

        wShift = Weights{DataType::kFLOAT, shiftArr, 1};
        wScale = Weights{DataType::kFLOAT, scaleArr, 1};
        wPower = Weights{DataType::kFLOAT, powerArr, 1};

        returnVal = network.addScale(*input, ScaleMode::kUNIFORM, wShift, wScale, wPower);
        std::string layerName = msg.name() + std::string("/reductionLayer/scaleLayer");
        returnVal->setName(layerName.c_str());
    }

    return returnVal;
}

ILayer* parseReshape(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::ReshapeParameter& p = msg.reshape_param();
    Dims bottomDims = tensors[msg.bottom(0)]->getDimensions();
    int axis = p.has_axis() ? p.axis() : 0;

    const ::ditcaffe::BlobShape& shape = p.shape();
    // Check that N (batch dim) is 0. TensorRT does not support reshape in batch dimension
    if ((axis == 0) && (shape.dim(0) != 0))
    {
        std::cout << "Caffe Parser: Invalid reshape param. TensorRT does not support reshape in N (batch) dimension" << std::endl;
        return nullptr;
    }

    // Handle axis and dims parameters
    int axStart = std::max(0, axis - 1);
    int axEnd = p.has_num_axes() ? std::max(0, axis - 1 + p.num_axes()) : bottomDims.nbDims;
    std::vector<int> reshapeDims;

    reshapeDims.reserve(axStart);
    for (int i = 0; i < axStart; i++)
    {
        reshapeDims.push_back(bottomDims.d[i]);
    }

    for (int i = 0; i < shape.dim_size(); i++)
    {
        // skip first 0 (batch)
        if (axis == 0 && i == 0)
        {
            continue;
        }
        if (shape.dim(i) == 0)
        {
            // If there is no bottom dimension corresponding to the current axis, then the params are invalid
            assert(static_cast<int>(reshapeDims.size()) <= bottomDims.nbDims);
            reshapeDims.push_back(bottomDims.d[reshapeDims.size()]);
        }
        else
        {
            reshapeDims.push_back(shape.dim(i));
        }
    }

    for (int i = axEnd; i < bottomDims.nbDims; i++)
    {
        reshapeDims.push_back(bottomDims.d[i]);
    }

    Dims topDims{};
    topDims.nbDims = static_cast<int>(reshapeDims.size());
    for (int i = 0; i < topDims.nbDims; i++)
    {
        topDims.d[i] = reshapeDims[i];
    }

    // Check there is at most one -1, and handle such case
    int countMinusOne = 0;
    for (int i = 0; i < topDims.nbDims; i++)
    {
        if (topDims.d[i] == -1)
        {
            countMinusOne += 1;
            // Inferred dimension
            int64_t newDim = volume(bottomDims) / -volume(topDims);
            topDims.d[i] = newDim;
        }
    }

    if (countMinusOne > 1)
    {
        std::cout << "Caffe Parser: Invalid reshape param. At most one axis can be inferred from other dimensions" << std::endl;
        return nullptr;
    }

    auto layer = network.addShuffle(*tensors[msg.bottom(0)]);
    layer->setReshapeDimensions(topDims);
    return layer;
}

ILayer* parsePermute(INetworkDefinition& network, const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const dc::PermuteParameter& p = msg.permute_param();
    Dims bottomDims = tensors[msg.bottom(0)]->getDimensions();
    Dims topDims = tensors[msg.bottom(0)]->getDimensions();
    int nbDims = bottomDims.nbDims;

    std::vector<int> orders;
    std::vector<bool> knownOrders(nbDims + 1, false);
    bool orderAbort = (p.order(0) != 0); // First order must be 0 (batch dimension)
    for (int i = 0; i < p.order_size(); i++)
    {
        int order = p.order(i);
        orderAbort |= (order > nbDims) || (std::find(orders.begin(), orders.end(), order) != orders.end());
        orders.push_back(order);
        knownOrders[order] = true;
    }

    if (orderAbort)
    {
        std::cout << "Caffe Parser: Invalid permute param. TensorRT does not support permute in N (batch) dimension, and order index must be within the tensor dimensions. no duplicate order allowed." << std::endl;
        return nullptr;
    }

    // Keep the rest of the order
    for (int i = 0; i < nbDims; i++)
    {
        if (!knownOrders[i])
        {
            orders.push_back(i);
        }
    }

    // Remove the first order (batch)
    orders.erase(orders.begin());

    for (int i = 0; i < nbDims; i++)
    {
        topDims.d[i] = bottomDims.d[orders[i] - 1];
    }
    assert(volume(topDims) == volume(bottomDims));

    nvinfer1::Permutation permuteOrder;
    for (int i = 0; i < nbDims; i++)
    {
        permuteOrder.order[i] = orders[i] - 1;
    }

    auto permute = network.addShuffle(*tensors[msg.bottom(0)]);
    permute->setReshapeDimensions(topDims);
    permute->setFirstTranspose(permuteOrder);
    return permute;
}

typedef ILayer* (*LayerParseFn)(INetworkDefinition&, const dc::LayerParameter&, CaffeWeightFactory&, BlobNameToTensor&);

std::unordered_map<std::string, LayerParseFn> gParseTable{
    {"Convolution", parseConvolution},
    {"Pooling", parsePooling},
    {"InnerProduct", parseInnerProduct},
    {"ReLU", parseReLU},
    {"Softmax", parseSoftMax},
    {"SoftmaxWithLoss", parseSoftMax},
    {"LRN", parseLRN},
    {"Power", parsePower},
    {"Eltwise", parseEltwise},
    {"Concat", parseConcat},
    {"Deconvolution", parseDeconvolution},
    {"Sigmoid", parseSigmoid},
    {"TanH", parseTanH},
    {"BatchNorm", parseBatchNormalization},
    {"Scale", parseScale},
    {"Crop", parseCrop},
    {"Reduction", parseReduction},
    {"Reshape", parseReshape},
    {"Permute", parsePermute},
    {"ELU", parseELU},
    {"BNLL", parseBNLL},
    {"Clip", parseClip},
    {"AbsVal", parseAbsVal},
    {"PReLU", parsePReLU}};

class CaffeParser : public ICaffeParser
{
public:
    const IBlobNameToTensor* parse(const char* deploy,
                                   const char* model,
                                   nvinfer1::INetworkDefinition& network,
                                   nvinfer1::DataType weightType) override;

    const IBlobNameToTensor* parseBuffers(const char* deployBuffer,
                                          size_t deployLength,
                                          const char* modelBuffer,
                                          size_t modelLength,
                                          INetworkDefinition& network,
                                          DataType weightType) override;

    void setProtobufBufferSize(size_t size) override { mProtobufBufferSize = size; }
    void setPluginFactory(nvcaffeparser1::IPluginFactory* factory) override { mPluginFactory = factory; }
    void setPluginFactoryExt(nvcaffeparser1::IPluginFactoryExt* factory) override
    {
        mPluginFactory = factory;
        mPluginFactoryIsExt = true;
    }

    void setPluginFactoryV2(nvcaffeparser1::IPluginFactoryV2* factory) override { mPluginFactoryV2 = factory; }
    void setPluginNamespace(const char* libNamespace) override { mPluginNamespace = libNamespace; }
    IBinaryProtoBlob* parseBinaryProto(const char* fileName) override;
    void destroy() override { delete this; }
    void setErrorRecorder(nvinfer1::IErrorRecorder* recorder) override { (void)recorder; assert(!"TRT- Not implemented."); }
    nvinfer1::IErrorRecorder* getErrorRecorder() const override { assert(!"TRT- Not implemented."); return nullptr; }

private:
    ~CaffeParser() override;
    std::vector<PluginField> parseNormalizeParam(const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    std::vector<PluginField> parsePriorBoxParam(const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    std::vector<PluginField> parseDetectionOutputParam(const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    std::vector<PluginField> parseLReLUParam(const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    std::vector<PluginField> parseRPROIParam(const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors);
    template <typename T>
    T* allocMemory(int size = 1)
    {
        T* tmpMem = static_cast<T*>(malloc(sizeof(T) * size));
        mTmpAllocs.push_back(tmpMem);
        return tmpMem;
    }

    const IBlobNameToTensor* parse(nvinfer1::INetworkDefinition& network,
                                   nvinfer1::DataType weightType,
                                   bool hasModel);

private:
    std::shared_ptr<ditcaffe::NetParameter> mDeploy;
    std::shared_ptr<ditcaffe::NetParameter> mModel;
    std::vector<void*> mTmpAllocs;
    BlobNameToTensor* mBlobNameToTensor{nullptr};
    size_t mProtobufBufferSize{INT_MAX};
    nvcaffeparser1::IPluginFactory* mPluginFactory{nullptr};
    nvcaffeparser1::IPluginFactoryV2* mPluginFactoryV2{nullptr};
    bool mPluginFactoryIsExt{false};
    std::vector<IPluginV2*> mNewPlugins;
    std::unordered_map<std::string, IPluginCreator*> mPluginRegistry;
    std::string mPluginNamespace = "";
};

extern "C" void* createNvCaffeParser_INTERNAL()
{
    return nvcaffeparser1::createCaffeParser();
}

ICaffeParser* nvcaffeparser1::createCaffeParser()
{
    return new CaffeParser;
}

CaffeParser::~CaffeParser()
{
    for (auto v : mTmpAllocs)
    {
        free(v);
    }
    for (auto p : mNewPlugins)
    {
        if (p)
        {
            p->destroy();
        }
    }
    delete mBlobNameToTensor;
}

std::vector<PluginField> CaffeParser::parseNormalizeParam(const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    std::vector<PluginField> f;
    const dc::NormalizeParameter& p = msg.norm_param();

    int* acrossSpatial = allocMemory<int32_t>();
    *acrossSpatial = p.across_spatial() ? 1 : 0;
    f.emplace_back("acrossSpatial", acrossSpatial, PluginFieldType::kINT32, 1);

    int* channelShared = allocMemory<int32_t>();
    *channelShared = p.channel_shared() ? 1 : 0;
    f.emplace_back("channelShared", channelShared, PluginFieldType::kINT32, 1);

    auto* eps = allocMemory<float>();
    *eps = p.eps();
    f.emplace_back("eps", eps, PluginFieldType::kFLOAT32, 1);

    std::vector<Weights> w;
    // If .caffemodel is not provided, need to randomize the weight
    if (!weightFactory.isInitialized())
    {
        int C = getCHW(tensors[msg.bottom(0)]->getDimensions()).c();
        w.emplace_back(weightFactory.allocateWeights(C, std::normal_distribution<float>(0.0F, 1.0F)));
    }
    else
    {
        // Use the provided weight from .caffemodel
        w = weightFactory.getAllWeights(msg.name());
    }

    for (auto weight : w)
    {
        f.emplace_back("weights", weight.values, PluginFieldType::kFLOAT32, weight.count);
    }

    int* nbWeights = allocMemory<int32_t>();
    *nbWeights = w.size();
    f.emplace_back("nbWeights", nbWeights, PluginFieldType::kINT32, 1);

    return f;
}

std::vector<PluginField> CaffeParser::parsePriorBoxParam(const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& /*tensors*/)
{
    std::vector<PluginField> f;
    const dc::PriorBoxParameter& p = msg.prior_box_param();

    int minSizeSize = p.min_size_size();
    auto* minSize = allocMemory<float>(minSizeSize);
    for (int i = 0; i < minSizeSize; ++i)
    {
        minSize[i] = p.min_size(i);
    }
    f.emplace_back("minSize", minSize, PluginFieldType::kFLOAT32, minSizeSize);

    int maxSizeSize = p.max_size_size();
    auto* maxSize = allocMemory<float>(maxSizeSize);
    for (int i = 0; i < maxSizeSize; ++i)
    {
        maxSize[i] = p.max_size(i);
    }
    f.emplace_back("maxSize", maxSize, PluginFieldType::kFLOAT32, maxSizeSize);

    int aspectRatiosSize = p.aspect_ratio_size();
    auto* aspectRatios = allocMemory<float>(aspectRatiosSize);
    for (int i = 0; i < aspectRatiosSize; ++i)
    {
        aspectRatios[i] = p.aspect_ratio(i);
    }
    f.emplace_back("aspectRatios", aspectRatios, PluginFieldType::kFLOAT32, aspectRatiosSize);

    int varianceSize = p.variance_size();
    auto* variance = allocMemory<float>(varianceSize);
    for (int i = 0; i < varianceSize; ++i)
    {
        variance[i] = p.variance(i);
    }
    f.emplace_back("variance", variance, PluginFieldType::kFLOAT32, varianceSize);

    int* flip = allocMemory<int32_t>();
    *flip = p.flip() ? 1 : 0;
    f.emplace_back("flip", flip, PluginFieldType::kINT32, 1);

    int* clip = allocMemory<int32_t>();
    *clip = p.clip() ? 1 : 0;
    f.emplace_back("clip", clip, PluginFieldType::kINT32, 1);

    int* imgH = allocMemory<int32_t>();
    *imgH = p.has_img_h() ? p.img_h() : p.img_size();
    f.emplace_back("imgH", imgH, PluginFieldType::kINT32, 1);

    int* imgW = allocMemory<int32_t>();
    *imgW = p.has_img_w() ? p.img_w() : p.img_size();
    f.emplace_back("imgW", imgW, PluginFieldType::kINT32, 1);

    auto* stepH = allocMemory<float>();
    *stepH = p.has_step_h() ? p.step_h() : p.step();
    f.emplace_back("stepH", stepH, PluginFieldType::kFLOAT32, 1);

    auto* stepW = allocMemory<float>();
    *stepW = p.has_step_w() ? p.step_w() : p.step();
    f.emplace_back("stepW", stepW, PluginFieldType::kFLOAT32, 1);

    auto* offset = allocMemory<float>();
    *offset = p.offset();
    f.emplace_back("offset", offset, PluginFieldType::kFLOAT32, 1);

    return f;
}

std::vector<PluginField> CaffeParser::parseDetectionOutputParam(const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& /*tensors*/)
{
    std::vector<PluginField> f;
    const dc::DetectionOutputParameter& p = msg.detection_output_param();
    const dc::NonMaximumSuppressionParameter& nmsp = p.nms_param();

    int* shareLocation = allocMemory<int32_t>();
    *shareLocation = p.share_location() ? 1 : 0;
    f.emplace_back("shareLocation", shareLocation, PluginFieldType::kINT32, 1);

    int* varianceEncodedInTarget = allocMemory<int32_t>();
    *varianceEncodedInTarget = p.variance_encoded_in_target() ? 1 : 0;
    f.emplace_back("varianceEncodedInTarget", varianceEncodedInTarget, PluginFieldType::kINT32, 1);

    int* backgroundLabelId = allocMemory<int32_t>();
    *backgroundLabelId = p.background_label_id();
    f.emplace_back("backgroundLabelId", backgroundLabelId, PluginFieldType::kINT32, 1);

    int* numClasses = allocMemory<int32_t>();
    *numClasses = p.num_classes();
    f.emplace_back("numClasses", numClasses, PluginFieldType::kINT32, 1);

    //nms
    int* topK = allocMemory<int32_t>();
    *topK = nmsp.top_k();
    f.emplace_back("topK", topK, PluginFieldType::kINT32, 1);

    int* keepTopK = allocMemory<int32_t>();
    *keepTopK = p.keep_top_k();
    f.emplace_back("keepTopK", keepTopK, PluginFieldType::kINT32, 1);

    auto* confidenceThreshold = allocMemory<float>();
    *confidenceThreshold = p.confidence_threshold();
    f.emplace_back("confidenceThreshold", confidenceThreshold, PluginFieldType::kFLOAT32, 1);

    //nms
    auto* nmsThreshold = allocMemory<float>();
    *nmsThreshold = nmsp.nms_threshold();
    f.emplace_back("nmsThreshold", nmsThreshold, PluginFieldType::kFLOAT32, 1);

    // input order = {0, 1, 2} in Caffe
    int* inputOrder = allocMemory<int32_t>(3);
    inputOrder[0] = 0;
    inputOrder[1] = 1;
    inputOrder[2] = 2;
    f.emplace_back("inputOrder", inputOrder, PluginFieldType::kINT32, 3);

    // confSigmoid = false for Caffe
    int* confSigmoid = allocMemory<int32_t>();
    *confSigmoid = 0;
    f.emplace_back("confSigmoid", confSigmoid, PluginFieldType::kINT32, 1);

    // isNormalized = true for Caffe
    int* isNormalized = allocMemory<int32_t>();
    *isNormalized = 1;
    f.emplace_back("isNormalized", isNormalized, PluginFieldType::kINT32, 1);

    // codeTypeSSD : from NvInferPlugin.h
    // CORNER = 0, CENTER_SIZE = 1, CORNER_SIZE = 2, TF_CENTER = 3
    int* codeType = allocMemory<int32_t>();
    switch (p.code_type())
    {
    case dc::PriorBoxParameter::CORNER_SIZE:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CORNER_SIZE);
        break;
    case dc::PriorBoxParameter::CENTER_SIZE:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CENTER_SIZE);
        break;
    case dc::PriorBoxParameter::CORNER: // CORNER is default
    default:
        *codeType = static_cast<int>(plugin::CodeTypeSSD::CORNER);
        break;
    }
    f.emplace_back("codeType", codeType, PluginFieldType::kINT32, 1);

    return f;
}

std::vector<PluginField> CaffeParser::parseLReLUParam(const dc::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& /*tensors*/)
{
    std::vector<PluginField> f;
    const dc::ReLUParameter& p = msg.relu_param();
    auto* negSlope = allocMemory<float>();
    *negSlope = p.negative_slope();
    f.emplace_back("negSlope", negSlope, PluginFieldType::kFLOAT32, 1);
    return f;
}
std::vector<PluginField> CaffeParser::parseRPROIParam(const dc::LayerParameter& msg, CaffeWeightFactory& weightFactory, BlobNameToTensor& tensors)
{
    std::vector<PluginField> f;
    const dc::ROIPoolingParameter& p1 = msg.roi_pooling_param();
    const dc::RegionProposalParameter& p2 = msg.region_proposal_param();

    // Memory allocations for plugin field variables
    int* poolingH = allocMemory<int32_t>();
    int* poolingW = allocMemory<int32_t>();
    auto* spatialScale = allocMemory<float>();
    int* preNmsTop = allocMemory<int32_t>();
    int* nmsMaxOut = allocMemory<int32_t>();
    auto* iouThreshold = allocMemory<float>();
    auto* minBoxSize = allocMemory<float>();
    int* featureStride = allocMemory<int32_t>();
    int* anchorsRatioCount = allocMemory<int32_t>();
    int* anchorsScaleCount = allocMemory<int32_t>();
    int anchorsRatiosSize = p2.anchor_ratio_size();
    auto* anchorsRatios = allocMemory<float>(anchorsRatiosSize);
    int anchorsScalesSize = p2.anchor_scale_size();
    auto* anchorsScales = allocMemory<float>(anchorsScalesSize);

    // Intialize the plugin fields with values from the prototxt
    *poolingH = p1.pooled_h();
    f.emplace_back("poolingH", poolingH, PluginFieldType::kINT32, 1);

    *poolingW = p1.pooled_w();
    f.emplace_back("poolingW", poolingW, PluginFieldType::kINT32, 1);

    *spatialScale = p1.spatial_scale();
    f.emplace_back("spatialScale", spatialScale, PluginFieldType::kFLOAT32, 1);

    *preNmsTop = p2.prenms_top();
    f.emplace_back("preNmsTop", preNmsTop, PluginFieldType::kINT32, 1);

    *nmsMaxOut = p2.nms_max_out();
    f.emplace_back("nmsMaxOut", nmsMaxOut, PluginFieldType::kINT32, 1);

    *iouThreshold = p2.iou_threshold();
    f.emplace_back("iouThreshold", iouThreshold, PluginFieldType::kFLOAT32, 1);

    *minBoxSize = p2.min_box_size();
    f.emplace_back("minBoxSize", minBoxSize, PluginFieldType::kFLOAT32, 1);

    *featureStride = p2.feature_stride();
    f.emplace_back("featureStride", featureStride, PluginFieldType::kINT32, 1);

    *anchorsRatioCount = p2.anchor_ratio_count();
    f.emplace_back("anchorsRatioCount", anchorsRatioCount, PluginFieldType::kINT32, 1);

    *anchorsScaleCount = p2.anchor_scale_count();
    f.emplace_back("anchorsScaleCount", anchorsScaleCount, PluginFieldType::kINT32, 1);

    for (int i = 0; i < anchorsRatiosSize; ++i) {
        anchorsRatios[i] = p2.anchor_ratio(i);
}
    f.emplace_back("anchorsRatios", anchorsRatios, PluginFieldType::kFLOAT32, anchorsRatiosSize);

    for (int i = 0; i < anchorsScalesSize; ++i) {
        anchorsScales[i] = p2.anchor_scale(i);
}
    f.emplace_back("anchorsScales", anchorsScales, PluginFieldType::kFLOAT32, anchorsScalesSize);

    return f;
}

const IBlobNameToTensor* CaffeParser::parseBuffers(const char* deployBuffer,
                                                   std::size_t deployLength,
                                                   const char* modelBuffer,
                                                   std::size_t modelLength,
                                                   INetworkDefinition& network,
                                                   DataType weightType)
{
    mDeploy = std::unique_ptr<dc::NetParameter>(new dc::NetParameter);
    google::protobuf::io::ArrayInputStream deployStream(deployBuffer, deployLength);
    if (!google::protobuf::TextFormat::Parse(&deployStream, mDeploy.get()))
    {
        RETURN_AND_LOG_ERROR(nullptr, "Could not parse deploy file");
    }

    if (modelBuffer)
    {
        mModel = std::unique_ptr<dc::NetParameter>(new dc::NetParameter);
        google::protobuf::io::ArrayInputStream modelStream(modelBuffer, modelLength);
        google::protobuf::io::CodedInputStream codedModelStream(&modelStream);
        codedModelStream.SetTotalBytesLimit(modelLength, -1);

        if (!mModel->ParseFromCodedStream(&codedModelStream))
        {
            RETURN_AND_LOG_ERROR(nullptr, "Could not parse model file");
        }
    }

    return parse(network, weightType, modelBuffer != nullptr);
}

const IBlobNameToTensor* CaffeParser::parse(const char* deployFile,
                                            const char* modelFile,
                                            INetworkDefinition& network,
                                            DataType weightType)
{
    CHECK_NULL_RET_NULL(deployFile)

    // this is used to deal with dropout layers which have different input and output
    mModel = std::unique_ptr<dc::NetParameter>(new dc::NetParameter);
    if (modelFile && !readBinaryProto(mModel.get(), modelFile, mProtobufBufferSize))
    {
        RETURN_AND_LOG_ERROR(nullptr, "Could not parse model file");
    }

    mDeploy = std::unique_ptr<dc::NetParameter>(new dc::NetParameter);
    if (!readTextProto(mDeploy.get(), deployFile))
    {
        RETURN_AND_LOG_ERROR(nullptr, "Could not parse deploy file");
    }

    return parse(network, weightType, modelFile != nullptr);
}

const IBlobNameToTensor* CaffeParser::parse(INetworkDefinition& network,
                                            DataType weightType,
                                            bool hasModel)
{
    bool ok = true;
    CaffeWeightFactory weights(*mModel.get(), weightType, mTmpAllocs, hasModel);

    mBlobNameToTensor = new (BlobNameToTensor);

    // Get list of all available plugin creators
    int numCreators = 0;
    IPluginCreator* const* tmpList = getPluginRegistry()->getPluginCreatorList(&numCreators);
    for (int k = 0; k < numCreators; ++k)
    {
        if (!tmpList[k])
        {
            std::cout << "Plugin Creator for plugin " << k << " is a nullptr." << std::endl;
            continue;
        }
        std::string pluginName = tmpList[k]->getPluginName();
        mPluginRegistry[pluginName] = tmpList[k];
    }

    for (int i = 0; i < mDeploy->input_size(); i++)
    {
        Dims dims;
        if (network.hasImplicitBatchDimension())
        {
            if (mDeploy->input_shape_size())
            {
                dims = DimsCHW{(int) mDeploy->input_shape().Get(i).dim().Get(1), (int) mDeploy->input_shape().Get(i).dim().Get(2), (int) mDeploy->input_shape().Get(i).dim().Get(3)};
            }
            else
            { // deprecated, but still used in a lot of networks
                dims = DimsCHW{(int) mDeploy->input_dim().Get(i * 4 + 1), (int) mDeploy->input_dim().Get(i * 4 + 2), (int) mDeploy->input_dim().Get(i * 4 + 3)};
            }
        }
        else
        {
            std::cout << "Warning, setting batch size to 1. Update the dimension after parsing due to using explicit batch size." << std::endl;
            if (mDeploy->input_shape_size())
            {
                dims = DimsNCHW{1, (int) mDeploy->input_shape().Get(i).dim().Get(1), (int) mDeploy->input_shape().Get(i).dim().Get(2), (int) mDeploy->input_shape().Get(i).dim().Get(3)};
            }
            else
            { // deprecated, but still used in a lot of networks
                dims = DimsNCHW{1, (int) mDeploy->input_dim().Get(i * 4 + 1), (int) mDeploy->input_dim().Get(i * 4 + 2), (int) mDeploy->input_dim().Get(i * 4 + 3)};
            }
        }
        ITensor* tensor = network.addInput(mDeploy->input().Get(i).c_str(), DataType::kFLOAT, dims);
        (*mBlobNameToTensor)[mDeploy->input().Get(i)] = tensor;
    }

    for (int i = 0; i < mDeploy->layer_size() && ok; i++)
    {
        const dc::LayerParameter& layerMsg = mDeploy->layer(i);
        if (layerMsg.has_phase() && layerMsg.phase() == dc::TEST)
        {
            continue;
        }

        // If there is a inplace operation and the operation is
        // modifying the input, emit an error as
        for (int j = 0; ok && j < layerMsg.top_size(); ++j)
        {
            for (int k = 0; ok && k < layerMsg.bottom_size(); ++k)
            {
                if (layerMsg.top().Get(j) == layerMsg.bottom().Get(k))
                {
                    auto iter = mBlobNameToTensor->find(layerMsg.top().Get(j).c_str());
                    if (iter != nullptr && iter->isNetworkInput())
                    {
                        ok = false;
                        std::cout << "TensorRT does not support in-place operations on input tensors in a prototxt file." << std::endl;
                    }
                }
            }
        }

        // If there is a pluginFactory provided, use layer name matching to handle the plugin construction
        if (mPluginFactory && mPluginFactory->isPlugin(layerMsg.name().c_str()))
        {
            std::vector<Weights> w = weights.getAllWeights(layerMsg.name());
            IPlugin* plugin = mPluginFactory->createPlugin(layerMsg.name().c_str(), w.empty() ? nullptr : &w[0], w.size());
            std::vector<ITensor*> inputs;
            for (int i = 0, n = layerMsg.bottom_size(); i < n; i++)
            {
                inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(i)]);
            }

            bool isExt = mPluginFactoryIsExt && static_cast<IPluginFactoryExt*>(mPluginFactory)->isPluginExt(layerMsg.name().c_str());

            ILayer* layer = isExt ? network.addPluginExt(&inputs[0], int(inputs.size()), *static_cast<IPluginExt*>(plugin))
                                  : network.addPlugin(&inputs[0], int(inputs.size()), *plugin);

            layer->setName(layerMsg.name().c_str());
            if (plugin->getNbOutputs() != layerMsg.top_size())
            {
                std::cout << "Plugin layer output count is not equal to caffe output count" << std::endl;
                ok = false;
            }
            for (int i = 0, n = std::min(layer->getNbOutputs(), layerMsg.top_size()); i < n; i++)
            {
                (*mBlobNameToTensor)[layerMsg.top(i)] = layer->getOutput(i);
            }

            if (layer == nullptr)
            {
                std::cout << "error parsing layer type " << layerMsg.type() << " index " << i << std::endl;
                ok = false;
            }

            continue;
        }
        if (getInferLibVersion() >= 5000)
        {
            if (mPluginFactoryV2 && mPluginFactoryV2->isPluginV2(layerMsg.name().c_str()))
            {
                if (mPluginFactory)
                {
                    RETURN_AND_LOG_ERROR(nullptr, "Both IPluginFactory and IPluginFactoryV2 are set. If using TensorRT 5.0 or later, switch to IPluginFactoryV2");
                }
                std::vector<Weights> w = weights.getAllWeights(layerMsg.name());
                IPluginV2* plugin = mPluginFactoryV2->createPlugin(layerMsg.name().c_str(), w.empty() ? nullptr : &w[0], w.size(), mPluginNamespace.c_str());
                std::vector<ITensor*> inputs;
                for (int i = 0, n = layerMsg.bottom_size(); i < n; i++)
                {
                    inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(i)]);
                }
                ILayer* layer = network.addPluginV2(&inputs[0], int(inputs.size()), *plugin);
                layer->setName(layerMsg.name().c_str());
                if (plugin->getNbOutputs() != layerMsg.top_size())
                {
                    std::cout << "Plugin layer output count is not equal to caffe output count" << std::endl;
                    ok = false;
                }
                for (int i = 0, n = std::min(layer->getNbOutputs(), layerMsg.top_size()); i < n; i++)
                {
                    (*mBlobNameToTensor)[layerMsg.top(i)] = layer->getOutput(i);
                }

                if (layer == nullptr)
                {
                    std::cout << "error parsing layer type " << layerMsg.type() << " index " << i << std::endl;
                    ok = false;
                }
                continue;
            }
            // Use the TRT5 plugin creator method to check for built-in plugin support


                std::string pluginName;
                PluginFieldCollection fc;
                std::vector<PluginField> f;
                if (layerMsg.type() == "Normalize")
                {
                    pluginName = "Normalize_TRT";
                    f = parseNormalizeParam(layerMsg, weights, *mBlobNameToTensor);
                }
                else if (layerMsg.type() == "PriorBox")
                {
                    pluginName = "PriorBox_TRT";
                    f = parsePriorBoxParam(layerMsg, weights, *mBlobNameToTensor);
                }
                else if (layerMsg.type() == "DetectionOutput")
                {
                    pluginName = "NMS_TRT";
                    f = parseDetectionOutputParam(layerMsg, weights, *mBlobNameToTensor);
                }
                else if (layerMsg.type() == "RPROI")
                {
                    pluginName = "RPROI_TRT";
                    f = parseRPROIParam(layerMsg, weights, *mBlobNameToTensor);
                }

                if (mPluginRegistry.find(pluginName) != mPluginRegistry.end())
                {
                    // Set fc
                    fc.nbFields = f.size();
                    fc.fields = f.empty() ? nullptr : f.data();
                    IPluginV2* pluginV2 = mPluginRegistry.at(pluginName)->createPlugin(layerMsg.name().c_str(), &fc);
                    assert(pluginV2);
                    mNewPlugins.push_back(pluginV2);

                    std::vector<ITensor*> inputs;
                    for (int i = 0, n = layerMsg.bottom_size(); i < n; i++)
                    {
                        inputs.push_back((*mBlobNameToTensor)[layerMsg.bottom(i)]);
                    }

                    auto layer = network.addPluginV2(&inputs[0], int(inputs.size()), *pluginV2);
                    layer->setName(layerMsg.name().c_str());
                    if (pluginV2->getNbOutputs() != layerMsg.top_size())
                    {
                        std::cout << "Plugin layer output count is not equal to caffe output count" << std::endl;
                        ok = false;
                    }
                    for (int i = 0, n = std::min(layer->getNbOutputs(), layerMsg.top_size()); i < n; i++)
                    {
                        (*mBlobNameToTensor)[layerMsg.top(i)] = layer->getOutput(i);
                    }

                    if (layer == nullptr)
                    {
                        std::cout << "error parsing layer type " << layerMsg.type() << " index " << i << std::endl;
                        ok = false;
                    }
                    continue;
                }

        }

        if (layerMsg.type() == "Dropout")
        {
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            continue;
        }

        if (layerMsg.type() == "Input")
        {
            const dc::InputParameter& p = layerMsg.input_param();
            for (int i = 0; i < layerMsg.top_size(); i++)
            {
                const dc::BlobShape& shape = p.shape().Get(i);
                if (shape.dim_size() != 4)
                {
                    RETURN_AND_LOG_ERROR(nullptr, "error parsing input layer, TensorRT only supports 4 dimensional input");
                }
                else
                {
                    DimsCHW dims{(int) shape.dim().Get(1), (int) shape.dim().Get(2), (int) shape.dim().Get(3)};
                    ITensor* tensor = network.addInput(layerMsg.top(i).c_str(), DataType::kFLOAT, dims);
                    (*mBlobNameToTensor)[layerMsg.top().Get(i)] = tensor;
                }
            }
            continue;
        }
        if (layerMsg.type() == "Flatten")
        {
            ITensor* tensor = (*mBlobNameToTensor)[layerMsg.bottom().Get(0)];
            (*mBlobNameToTensor)[layerMsg.top().Get(0)] = tensor;
            std::cout << "Warning: Flatten layer ignored. TensorRT implicitly"
                         " flattens input to FullyConnected layers, but in other"
                         " circumstances this will result in undefined behavior."
                      << std::endl;
            continue;
        }

        // Use parser table to lookup the corresponding parse function to handle the rest of the layers
        auto v = gParseTable.find(layerMsg.type());

        if (v == gParseTable.end())
        {
            std::cout << "could not parse layer type " << layerMsg.type() << std::endl;
            ok = false;
        }
        else
        {
            ILayer* layer = (*v->second)(network, layerMsg, weights, *static_cast<BlobNameToTensor*>(mBlobNameToTensor));
            if (layer == nullptr)
            {
                std::cout << "error parsing layer type " << layerMsg.type() << " index " << i << std::endl;
                ok = false;
            }
            else
            {
                layer->setName(layerMsg.name().c_str());
                (*mBlobNameToTensor)[layerMsg.top(0)] = layer->getOutput(0);
            }
        }
    }

    mBlobNameToTensor->setTensorNames();

    return ok && weights.isOK() && mBlobNameToTensor->isOK() ? mBlobNameToTensor : nullptr;
}

class BinaryProtoBlob : public IBinaryProtoBlob
{
public:
    BinaryProtoBlob(void* memory, DataType type, DimsNCHW dimensions)
        : mMemory(memory)
        , mDataType(type)
        , mDimensions(dimensions)
    {
    }

    DimsNCHW getDimensions() override
    {
        return mDimensions;
    }

    nvinfer1::DataType getDataType() override
    {
        return mDataType;
    }

    const void* getData() override
    {
        return mMemory;
    }

    void destroy() override
    {
        delete this;
    }

    ~BinaryProtoBlob() override
    {
        free(mMemory);
    }

    void* mMemory;
    DataType mDataType;
    DimsNCHW mDimensions;
};

IBinaryProtoBlob* CaffeParser::parseBinaryProto(const char* fileName)
{
    CHECK_NULL_RET_NULL(fileName)
    using namespace google::protobuf::io;

    std::ifstream stream(fileName, std::ios::in | std::ios::binary);
    if (!stream)
    {
        RETURN_AND_LOG_ERROR(nullptr, "Could not open file " + std::string{fileName});
    }

    IstreamInputStream rawInput(&stream);
    CodedInputStream codedInput(&rawInput);
    codedInput.SetTotalBytesLimit(INT_MAX, -1);

    ditcaffe::BlobProto blob;
    bool ok = blob.ParseFromCodedStream(&codedInput);
    stream.close();

    if (!ok)
    {
        RETURN_AND_LOG_ERROR(nullptr, "parseBinaryProto: Could not parse mean file");
    }

    DimsNCHW dims{1, 1, 1, 1};
    if (blob.has_shape())
    {
        int size = blob.shape().dim_size(), s[4] = {1, 1, 1, 1};
        for (int i = 4 - size; i < 4; i++)
        {
            assert(blob.shape().dim(i) < INT32_MAX);
            s[i] = static_cast<int>(blob.shape().dim(i));
        }
        dims = DimsNCHW{s[0], s[1], s[2], s[3]};
    }
    else
    {
        dims = DimsNCHW{blob.num(), blob.channels(), blob.height(), blob.width()};
    }

    const int dataSize = dims.n() * dims.c() * dims.h() * dims.w();
    assert(dataSize > 0);

    const dc::Type blobProtoDataType = CaffeWeightFactory::getBlobProtoDataType(blob);
    const auto blobProtoData = CaffeWeightFactory::getBlobProtoData(blob, blobProtoDataType, mTmpAllocs);

    if (dataSize != (int) blobProtoData.second)
    {
        std::cout << "CaffeParser::parseBinaryProto: blob dimensions don't match data size!!" << std::endl;
        return nullptr;
    }

    const int dataSizeBytes = dataSize * CaffeWeightFactory::sizeOfCaffeType(blobProtoDataType);
    void* memory = malloc(dataSizeBytes);
    memcpy(memory, blobProtoData.first, dataSizeBytes);
    return new BinaryProtoBlob(memory,
                               blobProtoDataType == ditcaffe::FLOAT ? DataType::kFLOAT : DataType::kHALF, dims);

    std::cout << "CaffeParser::parseBinaryProto: couldn't find any data!!" << std::endl;
    return nullptr;
}
