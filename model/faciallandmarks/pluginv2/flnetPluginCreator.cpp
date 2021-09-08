#include "softargmaxv2PluginV2Fl.hpp"
#include "upsamplingNearestPluginV2.hpp"

#include <vector>

namespace 
{
    const float FLNET_SOFTARGMAX_BETA = 0.1f;
    const float FLNET_SOFTARGMAX_EPSILON = 1e-6f;
    constexpr int FLNET_UPSAMPLING_FACTOR = 2;
}

// Strictly following the format of sampleUffSSD.cpp 
// even though the above sample code not following best C++ coding guide.
class SoftargmaxPluginCreator : public nvinfer1::IPluginCreator
{
public:
    SoftargmaxPluginCreator()
    {
        mFl.nbFields = 0;
        mFl.fields = 0;
    }

    const char* getPluginName() const override { return SOFTARGMAX_PLUGIN_TYPE.c_str(); }

    const char* getPluginVersion() const override { return SOFTARGMAX_PLUGIN_VERSION.c_str(); }

    const nvinfer1::PluginFieldCollection* getFieldNames() override { return &mFl; }

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fl) override
    {
        return new SoftargmaxPluginV2Fl(FLNET_SOFTARGMAX_BETA, FLNET_SOFTARGMAX_EPSILON);
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {
        //This object will be deleted when the network is destroyed, which will
        //call Concat::destroy()
        return new SoftargmaxPluginV2Fl(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    static nvinfer1::PluginFieldCollection mFl;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace = "";
};

nvinfer1::PluginFieldCollection SoftargmaxPluginCreator::mFl {};
std::vector<nvinfer1::PluginField> SoftargmaxPluginCreator::mPluginAttributes {};

REGISTER_TENSORRT_PLUGIN(SoftargmaxPluginCreator);

// Strictly following the format of sampleUffSSD.cpp 
// even though the above sample code not following best C++ coding guide.
class UpsamplingNearestPluginCreator : public nvinfer1::IPluginCreator
{
public:
    UpsamplingNearestPluginCreator()
    {
        mUN.nbFields = 0;
        mUN.fields = 0;
    }

    const char *getPluginName() const override { return UPSAMPLING_PLUGIN_TYPE.c_str(); }

    const char *getPluginVersion() const override { return UPSAMPLING_PLUGIN_VERSION.c_str(); }

    const nvinfer1::PluginFieldCollection* getFieldNames() override { return &mUN; }

    nvinfer1::IPluginV2* createPlugin(const char *name, const nvinfer1::PluginFieldCollection *fc) override
    {
        return new UpsamplingNearestPluginV2(FLNET_UPSAMPLING_FACTOR);
    }

    nvinfer1::IPluginV2* deserializePlugin(const char *name, const void *serialData, size_t serialLength) override
    {
        return new UpsamplingNearestPluginV2(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    static nvinfer1::PluginFieldCollection mUN;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace = "";
};

nvinfer1::PluginFieldCollection UpsamplingNearestPluginCreator::mUN {};
std::vector<nvinfer1::PluginField> UpsamplingNearestPluginCreator::mPluginAttributes {};

REGISTER_TENSORRT_PLUGIN(UpsamplingNearestPluginCreator);
