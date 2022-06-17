#!/bin/bash
set -eu
trap catch ERR
trap finally EXIT

CUDA_VER=10.2
GAZE_DIR=$(pwd)
DEEPSTREAM_DIR=/opt/nvidia/deepstream/deepstream/
WORKSPACE_SIZE=2000000000 
TAO_CONVERTER_URI=https://developer.nvidia.com/jp46-20210820t231431z-001zip

TEGRA_ID=`cat /sys/module/tegra_fuse/parameters/tegra_chip_id`

function catch {
	echo Setup failed. Please check error messages.
}
function finally {
	echo exit.
	cd $GAZE_DIR
}

# CHECK DEEPSTREAM INSTALATION
echo Checking DeepStream installation...
if [ ! -f $DEEPSTREAM_DIR/version ]; then 
	echo Installing DeepStream SDK...
	sudo apt install -y deepstream-6.0
fi
echo done.
echo

# INSTALL DEPENDENCIES
echo Installing dependencies...
sudo apt update
sudo apt install -y gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer1.0-dev\
	libgstreamer-plugins-base1.0-dev   libgstreamer-plugins-good1.0-dev   libgstreamer-plugins-bad1.0-dev\
	python3-dev python-gi-dev python3-pip git libgirepository1.0-dev libcairo2-dev apt-transport-https\
       	ca-certificates cmake libjpeg-dev
pip3 install Pillow azure-iot-device
echo done.
echo

# BUILD PYDS (DEEPSTREAM PYTHON BINDINGS)
echo Checking pyds installation...
set +e
trap - ERR
PYDS_AUTHOR=`pip3 show pyds | grep -i author:`
set -e
trap catch ERR
if [[ "$PYDS_AUTHOR" != *NVIDIA* ]]; then
	echo Building pyds...
	cd $GAZE_DIR/ds/lib
	if [ ! -d deepstream_python_apps ]; then
		git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps -b v1.1.1
	fi
	cd deepstream_python_apps/bindings/
	git submodule update --init
	mkdir build && cd build
	cmake ..  -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=6 -DPIP_PLATFORM=linux_aarch64 -DDS_PATH=$DEEPSTREAM_DIR
	make -j$(nproc)
	echo Installing pyds...
	pip3 install ./pyds-*.whl
	cd $GAZE_DIR
fi
echo done.
echo

# BUILD DS CLASSIFIER CUSTOM PARSER
echo Building classification cutom parser library...
cd $GAZE_DIR/ds/lib/customparser
env CUDA_VER=$CUDA_VER make
cp libcustomparser.so $GAZE_DIR/ds/lib
cd $GAZE_DIR
echo done.
echo

# BUILD DS GAZE INFER PLUGIN
echo Building Python wrapper library for gazeinfer...
cd $GAZE_DIR/ds/lib/gazeinfer
if [ ! -d deepstream_tao_apps ]; then
	git clone https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps -b release/tao3.0_ds6.0.1
fi

env CUDA_VER=$CUDA_VER make
cp dscprobes.so $GAZE_DIR/ds/lib
echo done.
echo
echo Building gazeinfer library...
cd deepstream_tao_apps/apps/tao_others/deepstream-gaze-app/gazeinfer_impl
env CUDA_VER=$CUDA_VER make
cp libnvds_gazeinfer.so $GAZE_DIR/ds/lib
cd $GAZE_DIR
echo done.
echo

# DOWNLOAD INFERENCE MODEL FROM NGC
cd $GAZE_DIR/model
echo Downloading face detection model... 
if [ ! -f face/facenet.etlt ]; then
	if [ "$TEGRA_ID" == 24 ] || [ "$TEGRA_ID" == 33 ]; then
		# fp16:TX2/TX1/Nano
		MODEL_URI=https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/deployable_v1.0/files/
		wget $MODEL_URI/model.etlt -O face/facenet.etlt
	else
		# int8:Xavier or later
		MODEL_URI=https://api.ngc.nvidia.com/v2/models/nvidia/tao/facenet/versions/pruned_quantized_v2.0.1/files/
		wget $MODEL_URI/model.etlt -O face/facenet.etlt
		wget $MODEL_URI/int8_calibration.txt -O face/facenet_cal.txt
	fi
fi
echo done.
echo
echo Downloading facial landmark model... 
if [ ! -f faciallandmarks/fpenet.etlt ]; then
	if [ "$TEGRA_ID" == 24 ] || [ "$TEGRA_ID" == 33 ]; then
		# fp16:TX2/TX1/Nano
		# FPENet model has different output layer names by either int8 or fp16 for now. (2022/05/01)
		# And FPENet postprocess codes in call_probes_from_py.cpp depend on the names.
		# So you need to use the model suitable for the codes.
		# Please see also the beggining part of ds/ds_pipeline.py.
		MODEL_URI=https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v1.0/files/
		wget $MODEL_URI/model.etlt -O faciallandmarks/fpenet.etlt
	else
		# int8:Xavier or later
		MODEL_URI=https://api.ngc.nvidia.com/v2/models/nvidia/tao/fpenet/versions/deployable_v3.0/files/
		wget $MODEL_URI/model.etlt -O faciallandmarks/fpenet.etlt
		wget $MODEL_URI/int8_calibration.txt -O faciallandmarks/fpenet_cal.txt
	fi
fi
echo done.
echo
echo Downloading gaze detection model... 
if [ ! -f gaze/gazenet_facegrid.etlt ]; then
	MODEL_URI=https://api.ngc.nvidia.com/v2/models/nvidia/tao/gazenet/versions/deployable_v1.0/files/
	wget $MODEL_URI/model.etlt -O gaze/gazenet_facegrid.etlt
fi
echo done.
cd $GAZE_DIR
echo

# BUILD GAZE
cd $GAZE_DIR/model/gaze
if [ ! -f $GAZE_DIR/model/gaze/tao-converter.zip ]; then
	echo Downloading tao-converter...
	wget $TAO_CONVERTER_URI -O tao-converter.zip
	echo done.
fi
unzip -jo tao-converter.zip '*/tao-converter'
if [ ! -f gazenet_facegrid_fp16_b8.engine ]; then
	echo Building gaze model...
	# tao-converter fails to convert when passed 4d(nchw) specs with multiple input model for now.
	# so give the additional dummy dimension(1xNxCxHxW) as a workaround. (@2022/04/28)
	./tao-converter -k nvidia_tlt -p input_face_images:0,1x1x224x224,4x1x224x224,8x1x224x224 \
	       			-p input_left_images:0,1x1x1x224x224,1x4x1x224x224,1x8x1x224x224 \
				-p input_right_images:0,1x1x1x224x224,1x4x1x224x224,1x8x1x224x224 \
				-p input_facegrid:0,1x1x1x625x1,1x4x1x625x1,1x8x1x625x1 \
				-m 8 -t fp16 -w $WORKSPACE_SIZE -e gazenet_facegrid_fp16_b8.engine \
				gazenet_facegrid.etlt
	echo done.
fi
cd $GAZE_DIR
echo

# CHANGE INFERENCE MODE
echo Setting inference mode INT8 or FP16
cd $GAZE_DIR/ds
chmod +x chinfmod.sh
if [ "$TEGRA_ID" == 24 ] || [ "$TEGRA_ID" == 33 ]; then
	# fp16:TX2/TX1/Nano
	./chinfmod.sh -fp16
else
	# int8:Xavier or later
	./chinfmod.sh -int8
fi
cd $GAZE_DIR
echo done.
echo

echo All done.
echo
