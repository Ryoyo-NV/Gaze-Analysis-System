#!/bin/bash

# for build gaze-analysis

unset DOCKER_BUILDKIT
docker build -t gaze-analysis .
