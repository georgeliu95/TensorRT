#include "regionPlugin.h"
#include "checkMacrosPlugin.h"
#include "yolo.h"
#include <cstring>

using namespace nvinfer1;
using nvinfer1::PluginType;
using nvinfer1::plugin::RegionPluginCreator;
using nvinfer1::plugin::Region;
using nvinfer1::plugin::RegionParameters; // Needed for Windows Build

namespace
{
const char* REGION_PLUGIN_VERSION{"1"};
const char* REGION_PLUGIN_NAME{"Region_TRT"};

template <typename T>
void safeFree(T* ptr)
{
    if (ptr)
    {
        free(ptr);
        ptr = nullptr;
    }
}

template <typename T>
void allocateChunk(T*& ptr, int count)
{
    ptr = static_cast<T*>(malloc(count * sizeof(T)));
}
}

PluginFieldCollection RegionPluginCreator::mFC{};
std::vector<PluginField> RegionPluginCreator::mPluginAttributes;

Region::Region(RegionParameters params)
    : num(params.num)
    , coords(params.coords)
    , classes(params.classes)
    , smTree(params.smTree)
{
}

Region::Region(const void* buffer, size_t length)
{
    const char *d = reinterpret_cast<const char *>(buffer), *a = d;
    C = read<int>(d);
    H = read<int>(d);
    W = read<int>(d);
    num = read<int>(d);
    classes = read<int>(d);
    coords = read<int>(d);
    bool softmaxTreePresent = read<bool>(d);
    bool leafPresent = read<bool>(d);
    bool parentPresent = read<bool>(d);
    bool childPresent = read<bool>(d);
    bool groupPresent = read<bool>(d);
    bool namePresent = read<bool>(d);
    bool groupSizePresent = read<bool>(d);
    bool groupOffsetPresent = read<bool>(d);
    if (softmaxTreePresent)
    {
        // need to read each element individually
        allocateChunk(smTree, 1);

        smTree->n = read<int>(d);

        if (leafPresent)
        {
            allocateChunk(smTree->leaf, smTree->n);
        }
        else
        {
            smTree->leaf = nullptr;
        }
        if (parentPresent)
        {
            allocateChunk(smTree->parent, smTree->n);
        }
        else
        {
            smTree->parent = nullptr;
        }
        if (childPresent)
        {
            allocateChunk(smTree->child, smTree->n);
        }
        else
        {
            smTree->child = nullptr;
        }
        if (groupPresent)
        {
            allocateChunk(smTree->group, smTree->n);
        }
        else
        {
            smTree->group = nullptr;
        }

        for (int i = 0; i < smTree->n; i++)
        {
            if (leafPresent)
            {
                smTree->leaf[i] = read<int>(d);
            }
            if (parentPresent)
            {
                smTree->parent[i] = read<int>(d);
            }
            if (childPresent)
            {
                smTree->child[i] = read<int>(d);
            }
            if (groupPresent)
            {
                smTree->group[i] = read<int>(d);
            }
        }

        if (namePresent)
        {
            allocateChunk(smTree->name, smTree->n);
        }
        else
        {
            smTree->name = nullptr;
        }

        if (namePresent)
        {
            for (int i = 0; i < smTree->n; i++)
            {
                allocateChunk(smTree->name[i], 256);
                for (int j = 0; j < 256; j++)
                {
                    smTree->name[i][j] = read<char>(d);
                }
            }
        }

        smTree->groups = read<int>(d);
        if (groupSizePresent)
        {
            allocateChunk(smTree->groupSize, smTree->groups);
        }
        else
        {
            smTree->groupSize = nullptr;
        }
        if (groupOffsetPresent)
        {
            allocateChunk(smTree->groupOffset, smTree->groups);
        }
        else
        {
            smTree->groupOffset = nullptr;
        }
        for (int i = 0; i < smTree->groups; i++)
        {
            if (groupSizePresent)
            {
                smTree->groupSize[i] = read<int>(d);
            }
            if (groupOffsetPresent)
            {
                smTree->groupOffset[i] = read<int>(d);
            }
        }
    }
    else
    {
        smTree = nullptr;
    }
    ASSERT(d == a + length);
}

int Region::getNbOutputs() const { return 1; }

Dims Region::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

int Region::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    const void* inputData = inputs[0];
    void* outputData = outputs[0];
    if (smTree)
    {
        hasSoftmaxTree = true;
    }
    else
    {
        hasSoftmaxTree = false;
    }
    yoloStatus_t status = regionInference(stream,
                                          batchSize, C, H, W,
                                          num, coords, classes,
                                          hasSoftmaxTree, smTree,
                                          inputData, outputData);
    ASSERT(status == STATUS_SUCCESS);
    return status;
}

size_t Region::getSerializationSize() const
{
    // C, H, W, num, classes, coords, smTree !nullptr and other array members !nullptr, softmaxTree members
    size_t count = 6 * sizeof(int) + 8 * sizeof(bool);
    if (smTree)
    {
        count += 2 * sizeof(int);

        if (smTree->leaf)
        {
            count += smTree->n * sizeof(int);
        }
        if (smTree->parent)
        {
            count += smTree->n * sizeof(int);
        }
        if (smTree->child)
        {
            count += smTree->n * sizeof(int);
        }
        if (smTree->group)
        {
            count += smTree->n * sizeof(int);
        }
        if (smTree->name)
        {
            count += smTree->n * 256 * sizeof(char);
        }
        if (smTree->groupSize)
        {
            count += smTree->groups * sizeof(int);
        }
        if (smTree->groupOffset)
        {
            count += smTree->groups * sizeof(int);
        }
    }
    return count;
}

void Region::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, C);
    write(d, H);
    write(d, W);
    write(d, num);
    write(d, classes);
    write(d, coords);
    write(d, smTree != nullptr);
    write(d, smTree != nullptr && smTree->leaf != nullptr);
    write(d, smTree != nullptr && smTree->parent != nullptr);
    write(d, smTree != nullptr && smTree->child != nullptr);
    write(d, smTree != nullptr && smTree->group != nullptr);
    write(d, smTree != nullptr && smTree->name != nullptr);
    write(d, smTree != nullptr && smTree->groupSize != nullptr);
    write(d, smTree != nullptr && smTree->groupOffset != nullptr);
    // need to do a deep copy
    if (smTree)
    {
        write(d, smTree->n);
        for (int i = 0; i < smTree->n; i++)
        {
            if (smTree->leaf)
            {
                write(d, smTree->leaf[i]);
            }
            if (smTree->parent)
            {
                write(d, smTree->parent[i]);
            }
            if (smTree->child)
            {
                write(d, smTree->child[i]);
            }
            if (smTree->group)
            {
                write(d, smTree->group[i]);
            }
        }
        if (smTree->name)
        {
            for (int i = 0; i < smTree->n; i++)
            {
                const char* str = smTree->name[i];
                for (int j = 0; j < 256; j++)
                {
                    write(d, str[j]);
                }
            }
        }
        write(d, smTree->groups);
        for (int i = 0; i < smTree->groups; i++)
        {
            if (smTree->groupSize)
            {
                write(d, smTree->groupSize[i]);
            }
            if (smTree->groupOffset)
            {
                write(d, smTree->groupOffset[i]);
            }
        }
    }
    ASSERT(d == a + getSerializationSize());
}

void Region::configureWithFormat(const Dims* inputDims, int nbInputs,
                                 const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int)
{
    ASSERT(type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);
    C = inputDims[0].d[0];
    H = inputDims[0].d[1];
    W = inputDims[0].d[2];
    ASSERT(C == num * (coords + 1 + classes));
}

bool Region::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

int Region::initialize() { return 0; }

void Region::terminate()
{
    // Do this carefully to guard against double frees
    if (smTree)
    {
        // free individual elements first
        safeFree(smTree->leaf);
        safeFree(smTree->parent);
        safeFree(smTree->child);
        safeFree(smTree->group);
        if (smTree->name)
        {
            for (int i = 0; i < smTree->n; i++)
            {
                safeFree(smTree->name[i]);
            }
            safeFree(smTree->name);
        }
        safeFree(smTree->groupSize);
        safeFree(smTree->groupOffset);

        // free softmax tree
        safeFree(smTree);
    }
}

const char* Region::getPluginType() const { return REGION_PLUGIN_NAME; }

const char* Region::getPluginVersion() const { return REGION_PLUGIN_VERSION; }

size_t Region::getWorkspaceSize(int maxBatchSize) const { return 0; }

void Region::destroy() { delete this; }

IPluginV2* Region::clone() const
{
    RegionParameters params{num, coords, classes, smTree};
    IPluginV2* plugin = new Region(params);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

RegionPluginCreator::RegionPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("coords", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("classes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("smTree", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RegionPluginCreator::getPluginName() const
{
    return REGION_PLUGIN_NAME;
}

const char* RegionPluginCreator::getPluginVersion() const
{
    return REGION_PLUGIN_VERSION;
}

const PluginFieldCollection* RegionPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2* RegionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "num"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.num = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "coords"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.coords = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "classes"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.classes = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "smTree"))
        {
            //TODO not sure if this will work
            void* tmpData = const_cast<void*>(fields[i].data);
            params.smTree = static_cast<nvinfer1::plugin::softmaxTree*>(tmpData);
        }
    }

    return new Region(params);
}

IPluginV2* RegionPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    //This object will be deleted when the network is destroyed, which will
    //call Region::destroy()
    return new Region(serialData, serialLength);
}
