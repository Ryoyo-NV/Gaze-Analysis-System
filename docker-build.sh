#!/bin/bash

mkdir sources ; pushd sources
cp -rf /opt/nvidia/deepstream/deepstream-6.0/sources/includes/    .
cp -rf /opt/nvidia/deepstream/deepstream-6.0/sources/gst-plugins/ .
popd

unset DOCKER_BUILDKIT
docker build -t gaze-analysis .
