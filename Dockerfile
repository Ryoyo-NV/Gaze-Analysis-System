FROM nvcr.io/nvidia/deepstream-l4t:6.0.1-base

WORKDIR /root/Gaze-Analysis-System
COPY sources/ /opt/nvidia/deepstream/deepstream-6.0/sources/
COPY . .

RUN env

RUN bash ./setup.sh 
