#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

// Custom classifier parser for gender estimation
// see also /opt/nvidie/deepstream/deepstream/sources/libs/nvdsinfer_customparser/nvdsinfer_customclassifierparser.cpp
extern "C"
bool NvDsInferClassifierParseGender (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString);

extern "C"
bool NvDsInferClassifierParseGender (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &descString)
{

    for (unsigned int i=0; i < outputLayersInfo.size(); ++i)
    {
//        NvDsInferDimsCHW dims;
//        getDimsCHWFromDims(dims, outputLayersInfo[i].inferDims);
//        std::cout << "output dims: " << dims.c << std::endl;

        float *outputCoverageBuffer = (float *)outputLayersInfo[i].buffer;
        NvDsInferAttribute attr;

        attr.attributeIndex = i;
        attr.attributeValue = 0;
	float confidence = outputCoverageBuffer[0];
	if (confidence < 0.)
	{
	    confidence = 0.;
	}
        attr.attributeConfidence = confidence;
        attr.attributeLabel = strdup("female");
	if (confidence > 0.5)
	{
            attr.attributeLabel = strdup("male");
	}
        attrList.push_back(attr);
        descString.append(attr.attributeLabel).append(" ");
    }

    return true;
}

CHECK_CUSTOM_CLASSIFIER_PARSE_FUNC_PROTOTYPE(NvDsInferClassifierParseGender);
