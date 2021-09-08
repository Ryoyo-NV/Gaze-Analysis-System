# Source: TensorRT-5.1.5.0/samples/python/uff_ssd/CMakeLists.txt

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(FlnetSoftargmax LANGUAGES CXX)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long")
# Use C++11
set (CMAKE_CXX_STANDARD 11)

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
find_package(CUDA REQUIRED)

set_ifndef(TRT_LIB /usr/lib/x86_64-linux-gnu)
set_ifndef(TRT_INCLUDE /usr/include/x86_64-linux-gnu)
set_ifndef(CUDA_ROOT /usr/local/cuda)

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})

# cuBLAS, not needed
# find_library(_CUBLAS_LIB cublas HINTS ${CUDA_ROOT} PATH_SUFFIXES lib lib64)
# set_ifndef(CUBLAS_LIB ${_CUBLAS_LIB})

# CUDA include dir
find_path(_CUDA_INC_DIR cuda_runtime_api.h HINTS ${CUDA_ROOT} PATH_SUFFIXES include)
set_ifndef(CUDA_INC_DIR ${_CUDA_INC_DIR})

# -------- BUILDING --------
SET (CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
SET (PLUGIN_CUDA_LIB flnetsoftargmax)

SET (PLUGIN_CUDA_SOURCES
    softargmaxv2.cu
    upsamplingNearest.cu
    flnetPluginCreator.cpp
)

include_directories(${TRT_INCLUDE} ${CUDA_INC_DIR})

# sm_6x is for Pascal GPUs, sm_7x is for Volta GPUs
cuda_add_library(${PLUGIN_CUDA_LIB} SHARED ${PLUGIN_CUDA_SOURCES}
    OPTIONS -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72
    -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
)
target_link_libraries(${PLUGIN_CUDA_LIB} PRIVATE ${NVINFER_LIB})

# -------- INSTALL --------
# Output the shared library into parent folder
install(TARGETS ${PLUGIN_CUDA_LIB} LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../../../server_custom_layers)
