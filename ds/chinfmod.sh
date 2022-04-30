#!/bin/bash

# Config files need to change int8 or fp16
confs=(
	"ds_pgie_facedetect_config.txt"
	"ds_sgie_faciallandmarks_config.txt"
)

if [ $# -ne 1 ]; then
	echo Please specify -int8 or -fp16 for inference mode
	exit 1
fi

if [ "$1" == "-int8" ]; then
	MODE_STR="int8"
	MODE=1
	PREV_STR="fp16"
elif [ "$1" == "-fp16" ]; then
	MODE_STR="fp16"
	MODE=2
	PREV_STR="int8"
else
	echo Invalid argument \"$1\"
	echo Please specify -int8 or -fp16 for inference mode
	exit 1
fi

# Change the model engine(cache) path and inference mode
for f in "${confs[@]}" ; do
	sed -i -e "s/^model-engine-file=\(.*\)${PREV_STR}/model-engine-file=\1${MODE_STR}/g" $f
	sed -i -e "s/^network-mode=.*/network-mode=${MODE}/g" $f
done

# Change the FPENet inference mode to suit the layer names for the model and postprocess
sed -i -e "s/^INFER_FPENET_MODEL_TYPE.*=.*/INFER_FPENET_MODEL_TYPE=${MODE}/g" "ds_pipeline.py"
