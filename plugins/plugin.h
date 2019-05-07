#ifndef TRT_PLUGIN_H
#define TRT_PLUGIN_H
#include "NvInfer.h"
#include <string>
// ENUMS {{{
typedef enum {
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

typedef pluginStatus_t frcnnStatus_t;
typedef pluginStatus_t ssdStatus_t;
typedef pluginStatus_t yoloStatus_t;

namespace nvinfer1
{
namespace plugin
{

class BasePlugin : public IPluginV2
{
protected:
    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

    std::string mNamespace;
};

class BaseCreator : public IPluginCreator
{
public:
    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }
protected:
    std::string mNamespace;
};

template <typename T>
void write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template <typename T>
T read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}
} // namespace plugin
} // namespace nvinfer1

#endif // TRT_PLUGIN_H
