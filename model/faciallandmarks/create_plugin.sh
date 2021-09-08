#!/bin/bash
cd pluginv2
rm -rf build
mkdir build
cd build
cmake ..
make -j12 install