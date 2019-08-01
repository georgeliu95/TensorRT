#include <cuda.h>

#if CUDA_VERSION >= 10000 && INCLUDE_MMA_KERNELS

#include "singleStepLSTMPlugin.h"

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

using namespace nvinfer1;
using nvinfer1::plugin::SingleStepLSTMPlugin;
using nvinfer1::plugin::SingleStepLSTMPluginCreator;

REGISTER_TENSORRT_PLUGIN(SingleStepLSTMPluginCreator);

SingleStepLSTMPlugin::SingleStepLSTMPlugin(const PluginFieldCollection *fc) {
    int idx = 0;
    
    mNumLayers = *(int*)(fc->fields[idx].data);
    idx++;
    
    mHiddenSize = *(int*)(fc->fields[idx].data);
    idx++;
    
    mAttentionSize = *(int*)(fc->fields[idx].data);
    idx++;
    
    mBeamSize = *(int*)(fc->fields[idx].data);
    idx++;
    
    mDataType = *(nvinfer1::DataType*)(fc->fields[idx].data);
    idx++;
}

SingleStepLSTMPlugin::SingleStepLSTMPlugin(const void* data, size_t length) {
    const char *d = static_cast<const char*>(data), *a = d;
    read<int>(d, mNumLayers);
    read<int>(d, mHiddenSize);
    read<int>(d, mAttentionSize);
    read<int>(d, mInputSize);
    read<int>(d, mBeamSize);
    
    read<nvinfer1::DataType>(d, mDataType);
    
    assert(d == a + length);  
}

const char* SingleStepLSTMPlugin::getPluginType() const {
    return "SingleStepLSTMPlugin";
}

const char* SingleStepLSTMPlugin::getPluginVersion() const {
    return "1";
}

void SingleStepLSTMPlugin::setPluginNamespace(const char* libNamespace) {
    mNamespace = libNamespace;
}

const char* SingleStepLSTMPlugin::getPluginNamespace() const {
    return mNamespace.c_str();
}

void SingleStepLSTMPlugin::destroy() {
    delete this;
}

void SingleStepLSTMPlugin::setCUDAInfo(cudaStream_t mStreami, cudaStream_t mStreamh, cudaStream_t* mSplitKStreams, cudaEvent_t* mSplitKEvents, cublasHandle_t mCublas) {
    this->mStreami = mStreami;
    this->mStreamh = mStreamh;
    this->mSplitKStreams = mSplitKStreams;
    this->mSplitKEvents = mSplitKEvents;
    this->mCublas = mCublas;
}

IPluginV2Ext* SingleStepLSTMPlugin::clone() const {
    size_t sz = getSerializationSize();
    
    char *buff = (char*)malloc(getSerializationSize());
    
    serialize(buff);
   
    SingleStepLSTMPlugin* ret = new SingleStepLSTMPlugin(buff, sz);
    
    ret->setCUDAInfo(mStreami, mStreamh, mSplitKStreams, mSplitKEvents, mCublas);
    
    free(buff);
    
    return ret;
}

int SingleStepLSTMPlugin::getNbOutputs() const {
    return 1 + 2 * mNumLayers;
}

// TODO: No idea if this needs batch size. Actually, don't know what's expected at all.
Dims SingleStepLSTMPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) {
    assert(index >= 0 && index < this->getNbOutputs());
    
    // y/hy/cy are all hiddenSize * batch.
    return Dims3(inputs[0].d[0], 1, mHiddenSize);
}

// Only half for now
bool SingleStepLSTMPlugin::supportsFormat(DataType type, PluginFormat format) const { 
    return type == DataType::kHALF || type == DataType::kINT8;
}

void SingleStepLSTMPlugin::configurePlugin (const Dims *inputDims, int nbInputs, const Dims *outputDims, int nbOutputs, const DataType *inputTypes, const DataType *outputTypes, const bool *inputIsBroadcast, const bool *outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) {
    assert(*inputTypes == DataType::kHALF);
    mInputSize = inputDims[0].d[inputDims[0].nbDims - 1];
}

int SingleStepLSTMPlugin::initialize() {
    CHECK(cublasCreate(&mCublas));
    
    CHECK(cublasSetMathMode(mCublas, CUBLAS_TENSOR_OP_MATH));
    
    CHECK(cudaStreamCreateWithPriority(&mStreami, 0, -1));
    CHECK(cudaStreamCreate(&mStreamh));
    mSplitKStreams = (cudaStream_t*)malloc(NUM_SPLIT_K_STREAMS * sizeof(cudaStream_t));
    mSplitKEvents = (cudaEvent_t*)malloc(NUM_SPLIT_K_STREAMS * sizeof(cudaEvent_t));

    for (int i = 0; i < NUM_SPLIT_K_STREAMS; i++) {
        CHECK(cudaStreamCreateWithPriority(&mSplitKStreams[i], 0, -1));
    }        

    return 0;
}

void SingleStepLSTMPlugin::terminate() {
    if (mCublas) {            
        CHECK(cublasDestroy(mCublas));
        mCublas = nullptr;
    }
    
    if (mStreami) {            
        CHECK(cudaStreamDestroy(mStreami));
        mStreami = nullptr;
    }
    if (mStreamh) {
        CHECK(cudaStreamDestroy(mStreamh));
        mStreamh = nullptr;
    }
            
    for (int i = 0; i < NUM_SPLIT_K_STREAMS; i++) {
        if (mSplitKStreams[i]) {               
            CHECK(cudaStreamDestroy(mSplitKStreams[i]));
            mSplitKStreams[i] = nullptr;
        }
    }

    if (mSplitKStreams) {           
        free(mSplitKStreams);
        mSplitKStreams = nullptr;
    }
    if (mSplitKEvents) {            
        free(mSplitKEvents);
        mSplitKEvents = nullptr;
    }
}

size_t SingleStepLSTMPlugin::getWorkspaceSize(int maxBatchSize) const {
    size_t size = 0;

    // tmp_io
    size += mNumLayers * (mAttentionSize + mInputSize) * maxBatchSize * mBeamSize * sizeof(half);
    
    // tmp_i
    size += mHiddenSize * maxBatchSize * mBeamSize * 4 * NUM_SPLIT_K_STREAMS * sizeof(half);
    
    // tmp_h
    size += mNumLayers * mHiddenSize * maxBatchSize * mBeamSize * 4 * sizeof(half);

    return size;
}

int SingleStepLSTMPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) {
    int effectiveBatch = batchSize * mBeamSize;

    assert(mAttentionSize == mHiddenSize);
    assert(mInputSize == mHiddenSize);

    void *tmp_io = workspace;
    void *tmp_i = (void*)((char*)(workspace) + (mNumLayers * mAttentionSize + mNumLayers * (mHiddenSize - 1) + mInputSize) * effectiveBatch * sizeof(half));
    void *tmp_h = (void*)((char*)(tmp_i) + mHiddenSize * effectiveBatch * 4 * NUM_SPLIT_K_STREAMS * sizeof(half));
    
    cudaEvent_t event;
    CHECK(cudaEventCreate(&event, cudaEventDisableTiming));
    CHECK(cudaEventRecord(event, stream));  
    CHECK(cudaStreamWaitEvent(mStreami, event, 0));
    CHECK(cudaStreamWaitEvent(mStreamh, event, 0));
    CHECK(cudaEventDestroy(event));  

    if (mDataType == nvinfer1::DataType::kHALF) {
        singleStepLSTMKernel<half, CUDA_R_16F, half, CUDA_R_16F>(mHiddenSize, 
                             mInputSize + mAttentionSize,
                             effectiveBatch, 
                             1,
                             mNumLayers,
                             this->mCublas,
                             (half*)inputs[0], // x 
                             (half**)(&(inputs[2])), // Array of hx, 
                             (half**)(&inputs[2 + mNumLayers]), // Array of cx, 
                             (half**)(&inputs[2 + 2 * mNumLayers]), // w, 
                             (half**)(&inputs[2 + 3 * mNumLayers]), // bias
                             (half*)outputs[0], // y, 
                             (half**)(&outputs[1]), // Array of hy, 
                             (half**)(&outputs[1 + mNumLayers]), // Array of cy,
                             (half*)inputs[1], // concatData,
                             (half*)tmp_io,
                             (half*)tmp_i,
                             (half*)tmp_h,
                             mStreami,
                             mSplitKStreams,
                             mSplitKEvents,
                             NUM_SPLIT_K_STREAMS,
                             mStreamh);
    }
             
    cudaEvent_t eventEnd;
    CHECK(cudaEventCreate(&eventEnd, cudaEventDisableTiming));
    CHECK(cudaEventRecord(eventEnd, mStreami));  
    CHECK(cudaStreamWaitEvent(stream, eventEnd, 0));
    CHECK(cudaEventDestroy(eventEnd));  
    
    return 0;
}

size_t SingleStepLSTMPlugin::getSerializationSize() const {
    size_t sz = sizeof(mNumLayers) + sizeof(mHiddenSize) + sizeof(mAttentionSize) + sizeof(mInputSize) + sizeof(mBeamSize) + sizeof(mDataType);
    return sz;
}

void SingleStepLSTMPlugin::serialize(void* buffer) const {
    char *d = static_cast<char*>(buffer), *a = d;

    write<int>(d, mNumLayers);
    write<int>(d, mHiddenSize);        
    write<int>(d, mAttentionSize);
    write<int>(d, mInputSize);
    write<int>(d, mBeamSize);
    write<nvinfer1::DataType>(d, mDataType);
    
    assert(d == a + getSerializationSize());
}

nvinfer1::DataType SingleStepLSTMPlugin::getOutputDataType (int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
    return mDataType;
}

bool SingleStepLSTMPlugin::isOutputBroadcastAcrossBatch (int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const {
    return false;
}

bool SingleStepLSTMPlugin::canBroadcastInputAcrossBatch (int inputIndex) const {
    return inputIndex >= 2 * mNumLayers + 2;
}

template <typename T>
void SingleStepLSTMPlugin::write(char*& buffer, const T& val) const
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
void SingleStepLSTMPlugin::read(const char*& buffer, T& val) const
{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
}

const char* SingleStepLSTMPluginCreator::getPluginName() const {
    return "SingleStepLSTMPlugin";
}

const char* SingleStepLSTMPluginCreator::getPluginVersion() const {
    return "1";
}

// Not sure why I need names. Can't do it with variable layer count anyway.
const PluginFieldCollection* SingleStepLSTMPluginCreator::getFieldNames() {
    return nullptr;        
}

void SingleStepLSTMPluginCreator::setPluginNamespace(const char* libNamespace) {
    mNamespace = libNamespace;
}

const char* SingleStepLSTMPluginCreator::getPluginNamespace() const {
    return mNamespace.c_str();
}

IPluginV2Ext* SingleStepLSTMPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) {
    return new SingleStepLSTMPlugin(fc);        
}

IPluginV2Ext* SingleStepLSTMPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) {
    return new SingleStepLSTMPlugin(serialData, serialLength);        
}

#endif /* CUDA_VERSION >= 10000 && INCLUDE_MMA_KERNELS */
