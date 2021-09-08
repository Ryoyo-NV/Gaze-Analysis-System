## How to build the flnet TensorRT engine with plugin?

`pluginv2/`
This directory contains files for plugin needed for flnet.

`softargmaxv2PluginV2Fl.hpp`
PluginV2 definition of softargmax used by Facial landmark (without depth output)

`upsamplingNearestPluginV2.hpp`
PluginV2 definition of upsampling nearest 

`flnetPluginCreator.cpp`
PluginCreator for the above two layers for them to be available in TensorRT PluginRegistry

## Prerequisites

1. [Install cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).

2. [Install CMake (minimum version 3.11)](https://cmake.org/download/).

3. [Download TensorRT](https://developer.nvidia.com/tensorrt).

## Creating and using the plugin

1.  Build the plugin
  ```
  cd pluginv2
  mkdir build && cd build
  cmake ..
  ```

  **Note:** If any of the dependencies are not installed in their default locations, you can manually specify them. For example:
  ```
  cmake .. 
  -DCUDA_ROOT=/usr/local/cuda-9.2/
  -DNVINFER_LIB=/path/to/libnvinfer.so -DTRT_INC_DIR=/path/to/tensorrt/include/
  ```

  `cmake ..` displays a complete list of configurable variables. If a variable is set to `VARIABLE_NAME-NOTFOUND`, then you’ll need to specify it manually or set the variable it is derived from correctly.
  ```
  make -j12 install
  ```
  a `libflnetsoftargmax.so` will be generated at the same folder of this README

2. Use the plugin in python
  ``` 
  import ctypes 
  ctypes.CDLL(os.path.join(path/to/libflnetsoftargmax.so))
  tensorrt.init_libnvinfer_plugins(TRT_LOGGER, '')
  ```

  Then the plugin will be automatically picked up from the uff parser

3. Use the plugin in C++

  Link the shared library with your binary or run your binary with `LD_PRELOAD=/path/to/libflnetsoftargmax.so`

  In your code, run `initLibNvInferPlugins(...)` before using the uff parser