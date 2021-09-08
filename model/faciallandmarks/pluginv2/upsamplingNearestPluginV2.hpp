#ifndef _UPSAMPLING_PLUGIN_V2_HPP_
#define _UPSAMPLING_PLUGIN_V2_HPP_

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cassert>

void deconv_nearest_neighbor_upsampling(cudaStream_t stream, void* d_output, const void* d_input,
    int N, int C, int H, int W, int upsamplingFactor, bool is_fp16);

const std::string UPSAMPLING_PLUGIN_TYPE = "UpsamplingNearest";
const std::string UPSAMPLING_PLUGIN_VERSION = "1";

class UpsamplingNearestPluginV2 : public nvinfer1::IPluginV2
{
public:
    UpsamplingNearestPluginV2(const int upsamplingFactor) 
        : mUpsamplingFactor(upsamplingFactor)
    {

    }

    UpsamplingNearestPluginV2(const void *data, size_t length)
    {
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        read(mInputDims, d);
        read(mDataType, d);
        read(mDataSize, d);
        read(mUpsamplingFactor, d);
        read(mMaxBatchSize, d);
        assert(d == a + length);
    }

    int getNbOutputs() const override { return 1; }

    nvinfer1::Dims getOutputDimensions(
        int index, const nvinfer1::Dims* inputs, int nbInputDims) override
    {
        assert((index == 0) && (nbInputDims == 1));
        nvinfer1::DimsCHW result = nvinfer1::DimsCHW(
            inputs[0].d[0], inputs[0].d[1] * mUpsamplingFactor, inputs[0].d[2] * mUpsamplingFactor);
        return result;
    }

    int initialize() override { return 0; }

    void terminate() override {}

    size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace,
        cudaStream_t stream) override
    {
        bool is_fp16 = (mDataType == nvinfer1::DataType::kFLOAT) ? false : true;
        deconv_nearest_neighbor_upsampling(stream, outputs[0], inputs[0], batchSize, mInputDims.c(),
            mInputDims.h(), mInputDims.w(), mUpsamplingFactor, is_fp16);
        return 0;
    }

    size_t getSerializationSize() const override
    {
        size_t result = 0;
        result += sizeof(mInputDims);
        result += sizeof(mDataType);
        result += sizeof(mDataSize);
        result += sizeof(mUpsamplingFactor);
        result += sizeof(mMaxBatchSize);
        return result;
    }

    void serialize(void* buffer) const override
    {
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, mInputDims);
        write(d, mDataType);
        write(d, mDataSize);
        write(d, mUpsamplingFactor);
        write(d, mMaxBatchSize);
        assert(d == a + getSerializationSize());
    }

    void configureWithFormat(const nvinfer1::Dims *inputDims, int nbInputs, const nvinfer1::Dims *outputDims,
        int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
    {
        assert(nbInputs == 1);
        assert(nbOutputs == 1);
        assert(format == nvinfer1::PluginFormat::kNCHW);
        assert((type == nvinfer1::DataType::kHALF) || type == (nvinfer1::DataType::kFLOAT));
        mInputDims = nvinfer1::DimsCHW(inputDims[0].d[0], inputDims[0].d[1], inputDims[0].d[2]);
        mDataType = type;
        mDataSize = (type == nvinfer1::DataType::kFLOAT) ? sizeof(float) : sizeof(__half);
        mMaxBatchSize = maxBatchSize;
    }

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override
    {
        return (type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT) 
           && format == nvinfer1::PluginFormat::kNCHW;
    }

    const char* getPluginType() const override
    {
        return UPSAMPLING_PLUGIN_TYPE.c_str();
    }

    const char* getPluginVersion() const override
    {
        return UPSAMPLING_PLUGIN_VERSION.c_str();
    }

    void destroy() override
    {
        delete this;
    }

    nvinfer1::IPluginV2* clone() const override
    {
        return new UpsamplingNearestPluginV2(mUpsamplingFactor);
    }

    void setPluginNamespace(const char* libNamespace) override
    {
        mNamespace = libNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mNamespace.c_str();
    }

private:
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

    nvinfer1::DimsCHW mInputDims;
    nvinfer1::DataType mDataType {nvinfer1::DataType::kHALF};
    size_t mDataSize {sizeof(__half)};
    size_t mMaxBatchSize {1};
    int mUpsamplingFactor {-1};
    std::string mNamespace {""};
};

#endif // _UPSAMPLING_PLUGIN_V2_HPP_