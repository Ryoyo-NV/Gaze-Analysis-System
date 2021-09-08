#ifndef _CUSTOMPLUTIN_UTILS_H
#define _CUSTOMPLUTIN_UTILS_H

#include <glog/logging.h>

#define CHECK_CUDA(status)                                         \
    {                                                              \
        if (status != 0) {                                         \
            LOG(ERROR) << "Cuda failure: " << status << std::endl; \
            abort();                                               \
        }                                                          \
    }

#endif /* _CUSTOMPLUTIN_UTILS_H */
