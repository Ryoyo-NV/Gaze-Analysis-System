#!/bin/bash

GAZE_DIR=$(pwd)
CUDA_VER=10.2

# INSTALL DEPENDENCIES
echo Installing dependencies...
sudo apt install -y gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer1.0-dev\
	libgstreamer-plugins-base1.0-dev   libgstreamer-plugins-good1.0-dev   libgstreamer-plugins-bad1.0-dev\
	python3-dev python-gi-dev python3-pip git libgirepository1.0-dev libcairo2-dev apt-transport-https\
       	ca-certificates cmake libjpeg-dev
pip3 install Pillow azure-iot-device

# BUILD PYDS (DEEPSTREAM PYTHON BINDINGS)
echo Building pyds...
cd $GAZE_DIR/ds/lib
git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
cd deepstream_python_apps/bindings/
git submodule update --init
mkdir build && cd build
cmake ..  -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=6 -DPIP_PLATFORM=linux_aarch64 -DDS_PATH=/opt/nvidia/deepstream/deepstream
make -j$(nproc)
pip3 install ./pyds-*.whl
cd $GAZE_DIR

# BUILD DS CLASSIFIER CUSTOM PARSER
echo Building classification cutom parser library...
cd $GAZE_DIR/ds/lib/customparser
env CUDA_VER=$CUDA_VER make
cp libcustomparser.so $GAZE_DIR/ds/lib
cd $GAZE_DIR

# BUILD DS GAZE INFER PLUGIN
echo Building gazeinfer library...
cd $GAZE_DIR/ds/lib/gazeinfer
git clone https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps
env CUDA_VER=$CUDA_VER make
cp dscprobe.so $GAZE_DIR/ds/lib

cd deepstream_tao_apps/apps/tao_others/deepstream-gaze-app/gazeinfer_impl
env CUDA_VER=$CUDA_VER make
cp libnvds_gazeinfer.so $GAZE_DIR/ds/lib
cd $GAZE_DIR

# DOWNLOAD INFERENCE MODEL FROM NGC
echo Downloading face detection model...
cd $GAZE_DIR/model
# for fp16
#wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/deployable_v1.0/files/model.etlt -O model/face/facenet.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/model.etlt -O face/facenet.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/int8_calibration.txt -O face/facenet_cal.txt
echo Downloading facial landmark model...
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v3.0/files/model.etlt -O faciallandmarks/fpenet.etlt
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v3.0/files/int8_calibration.txt -O faciallandmarks/fpenet_cal.txt
echo Downloading gaze detection model...
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/gazenet/versions/deployable_v1.0/files/model.etlt -O gaze/gazenet_facegrid.etlt
cd $GAZE_DIR

# BUILD GAZE
echo Downloading tao-converter...
cd $GAZE_DIR/model/gaze
wget https://developer.nvidia.com/jp46-20210820t231431z-001zip -O tao-converter-jp46.zip
unzip -j tao-converter-jp46.zip '*/tao-converter'
echo Building gaze model...
tao-converter -k nvidia_tlt -p input_face_images:0,1x1x224x224,4x1x224x224,8x1x224x224 -p input_left_images:0,1x1x1x224x224,1x4x1x224x224,1x8x1x224x224 -p input_right_images:0,1x1x1x224x224,1x4x1x224x224,1x8x1x224x224 -p input_facegrid:0,1x1x1x625x1,1x4x1x625x1,1x8x1x625x1 -m 8 -t fp16 -w 1500000000 -e gazenet_facegrid_fp16_b8.engine gazenet.etlt

