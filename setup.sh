#!/bin/bash

OUTPUT_DIR=$(pwd)
FACE_BATCH_SIZE=4

# BUILD FACIAL LANDMARKS PLUGIN
echo Building facial landmarks plugin...
cd model/faciallandmarks

#./create_plugin.sh
mkdir -p pluginv2/build
cd pluginv2/build
cmake .. -DTRT_LIB=/usr/lib/aarch64-linux-gnu -DTRT_INCLUDE=/usr/include/aarch64-linux-gnu
make -j
cd ../..

mv pluginv2/build/libflnetsoftargmax.so $OUTPUT_DIR
cd ../..

# BUILD DS CLASSIFIER CUSTOM PARSER
cd ds
make -j
cd ..

# BUILD FACIAL LANDMARKS MODEL
echo Building facial landmarks model...
cp model/faciallandmarks/1/model.hdf5.uff .
/usr/src/tensorrt/bin/trtexec \
  --uff=model.hdf5.uff \
  --uffInput=input_face_images,1,80,80 \
  --output=softargmax \
  --plugins=$OUTPUT_DIR/libflnetsoftargmax.so \
  --saveEngine=$OUTPUT_DIR/flm_model.engine \
  --fp16 \
  --maxBatch=$FACE_BATCH_SIZE

# BUILD GAZE
echo Building gaze model...
cp model/gaze/1/gaze.uff .
/usr/src/tensorrt/bin/trtexec \
  --uff=gaze.uff \
  --uffInput=input_left_images,1,224,224 \
  --uffInput=input_right_images,1,224,224 \
  --uffInput=input_face_images,1,224,224 \
  --uffInput=input_landmarks,1,136,1 \
  --output=fc_joint/concat \
  --saveEngine=$OUTPUT_DIR/gaze_model.engine \
  --fp16 \
  --maxBatch=$FACE_BATCH_SIZE
  #--useDLACore=0 \
  #--allowGPUFallback

