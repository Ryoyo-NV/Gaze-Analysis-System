#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

// Custom classifier parser for age estimation
// see also /opt/nvidie/deepstream/deepstream/sources/libs/nvdsinfer_customparser/nvdsinfer_customclassifierparser.cpp
extern "C"
bool NvDsInferClassifierParseAge (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString);

extern "C"
bool NvDsInferClassifierParseAge (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString)
{

    for (unsigned int i=0; i < outputLayersInfo.size(); ++i)
    {
        float *outputCoverageBuffer = (float *)outputLayersInfo[i].buffer;
        NvDsInferAttribute attr;
    
        attr.attributeIndex = i;
        attr.attributeValue = 0;
        attr.attributeConfidence = outputCoverageBuffer[0];
        attr.attributeLabel = "0";
        attrList.push_back(attr);
        descString.append(attr.attributeLabel).append(" ");
    }

    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassifierParseAge);
