#ifndef _SOFTARGMAX_PLUGIN_V2_HAND_HPP_
#define _SOFTARGMAX_PLUGIN_V2_HAND_HPP_

#include "fp16.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"

#include <cassert>
#include <string>

const std::string SOFTARGMAX_PLUGIN_TYPE = "Python";
const std::string SOFTARGMAX_PLUGIN_VERSION = "1";

void softargmax_kernel_hand(cudaStream_t stream, const void* const* inputs, void** outputs,
    const int N, const int C, const int H, const int W, float beta, float epsilon, bool isfp16);

// Note: the V2 interface has a *clone that breaks the inheritance pattern in IPluginExt
// The code between two kinds of plugin has to be duplicated
class SoftargmaxPluginV2Hand : public nvinfer1::IPluginV2
{
public:
    SoftargmaxPluginV2Hand(const float beta, const float epsilon)
        : mBeta(beta)
        , mEpsilon(epsilon)
    {
    }

    // Create the plugin at runtime from a byte stream
    SoftargmaxPluginV2Hand(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        read(mBeta, d);
        read(mEpsilon, d);
        read(mInputDims, d);
        read(mMaxBatchSize, d);
        read(mDataType, d);
        assert(d == a + length);
    }

    int initialize() override { return 0; }

    void terminate() override {}

    size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace,
        cudaStream_t stream) override
    {
        bool isfp16 = (mDataType == nvinfer1::DataType::kHALF);
        softargmax_kernel_hand(stream, inputs, outputs, 1, mInputDims.c() / 2, mInputDims.h(),
            mInputDims.w(), mBeta, mEpsilon, isfp16);
        return 0;
    }

    int getNbOutputs() const override
    {
        // (X, Y) and confidence
        return 2;
    }
 
    nvinfer1::Dims getOutputDimensions(
        int index, const nvinfer1::Dims* inputs, int nbInputDims) override
    {
        assert((index >= 0) && (index <= 1) && (nbInputDims == 1));
        nvinfer1::DimsCHW result;
        switch (index) {
        case 0:  // preds
            result = nvinfer1::DimsCHW((inputs[0].d[0] / 2 * 3), 1, 1);
            break;
        case 1:  // confidence
            result = nvinfer1::DimsCHW(inputs[0].d[0] / 2, 1, 1);
            break;
        default:
            break;
        }
        return result;
    }

    void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
        const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type,
        nvinfer1::PluginFormat format, int maxBatchSize) override
    {
        assert(nbInputs == 1);
        assert(nbOutputs == 2);
        mInputDims = nvinfer1::DimsCHW(inputDims[0].d[0], inputDims[0].d[1], inputDims[0].d[2]);
        mDataType = type;
        mMaxBatchSize = maxBatchSize;
    }

    size_t getSerializationSize() const override
    {
        size_t result = 0;
        result += sizeof(mBeta);
        result += sizeof(mEpsilon);
        result += sizeof(mInputDims);
        result += sizeof(mMaxBatchSize);
        result += sizeof(mDataType);
        return result;
    }

    void serialize(void* buffer) const override
    {
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, mBeta);
        write(d, mEpsilon);
        write(d, mInputDims);
        write(d, mMaxBatchSize);
        write(d, mDataType);
        assert(d == a + getSerializationSize());
    }

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override
    {
        return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF) &&
               format == nvinfer1::PluginFormat::kNCHW;
    }

    const char* getPluginType() const override { return SOFTARGMAX_PLUGIN_TYPE.c_str(); }

    const char* getPluginVersion() const override { return SOFTARGMAX_PLUGIN_VERSION.c_str(); }

    void destroy() override { delete this; }

    nvinfer1::IPluginV2* clone() const override { return new SoftargmaxPluginV2Hand(mBeta, mEpsilon); }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

protected:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    void read(T& val, const char*& buffer) const
    {
        val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
    }

    float mBeta {0.1f};
    float mEpsilon {1e-6f};
    nvinfer1::DimsCHW mInputDims;
    nvinfer1::DataType mDataType {nvinfer1::DataType::kHALF};
    size_t mMaxBatchSize {1};
    std::string mNamespace {""};
};


#endif // _SOFTARGMAX_PLUGIN_V2_HAND_HPP_
