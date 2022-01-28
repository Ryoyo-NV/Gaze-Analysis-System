# Gaze Detection System
This is a repository of  Gaze Detection System using NVIDIA 's Jetson and DeepStream 6.0.

For example, it is useful for measuring the effectiveness of signage advertising.  
From webcom or pre-made videos, it detect the gaze of person watching and analyze the gender and age of that person. 
Each person detected will be drawn on bounding box. It is also possible to visualize data by an authorized Azure Iot Hub and linking with Open Distro for Kibana. 


<img src="src/gaze_demo.gif" hight="480"/>  
  
 Gaze detection system used the content of 5 models:  
 The Face detection, Face landmarks, Gaze detection, Age estimation, and Gender estimation model. 


See [Reference of Content Models](model/README.md) out for details (including model's link).  
**Please note that the models are available for research, development, PoC purposes only.**  
For uses other than the above, please replace the model with another model.

## Prerequisite

- NVIDIA Jetson Platform
- [JetPack](https://developer.nvidia.com/embedded/jetpack) 4.6
- Video(.h264 format) or USB webcam

Test on:

- Jetson Xavier NX, JetPack 4.6, Video, and USB webcam. 

Note: Recommend Jetson Xavier NX or AGX Xavier.  
The Age and gender prebuild engine model with TenserRT are built for Xavier GPU(compute capability 7.2).  
If you want to run on not Xavier(e.g. Nano), you need to rebuild them on your device.
See [model/readme](https://github.com/Ryoyo-NV/Gaze-Analysis-System/tree/main/model), and more.

## Installation
There are two options to setup and it can be run local or Docker container.

### Requirements:  

- [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk) 6.0
- [PyTorch for JetPack 4.6](https://elinux.org/Jetson_Zoo) (PyTorch 1.6.0, Torchvision 0.6.0)
- Cython 0.29.21
- [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
- TensorRT 7.1.3
- PyCuda 2020.1
- Gstremer 1.0

Options (using visualize data):
- Open Distro for ElasticSearch 1.10.1
- Open Distro for Kibana 1.10.1
- Azure Iot Hub
- Azure Visual Machine

Note: Excludes packages already included in JetPack. (ex. Python3.6, OpenCV, and etc.)  
See above for tested versions. 
### Option 1: Installing local  

#### 1. Clone this repository 
```
$ git clone <this repo>
```
#### 2. To install PyTorch

```
#PyTorch 1.6.0
$ wget https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl -O torch-1.6.0-cp36-cp36m-linux_aarch64.whl
$ sudo apt install python3-pip libopenblas-base libopenmpi-dev
$ sudo pip3 install Cython
$ sudo pip3 install numpy torch-1.6.0-cp36-cp36m-linux_aarch64.whl

#Torchvision 0.6.0
$ sudo apt install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
$ git clone --branch v0.6.0 https://github.com/pytorch/vision torchvision
$ cd torchvision
$ sudo python3 setup.py install
```
See [PyTorch for JetPack 4.4]( https://elinux.org/Jetson_Zoo), and more.


#### 3. To install PyCuda
Set up the development environment by modifying the PATH and LD_LIBRARY_PATH variables:
```
$ sudo vim ~/.bashrc

# Add the PATH following
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
```
$ source ~/.bashrc
$ pip3 install pycuda
```

#### 4. To install DeepStream SDK
Installing the five methods. Using the DeepStream tar package here.  
Download package [here](https://developer.nvidia.com/deepstream-getting-started) .

```
$ sudo tar -xvf deepstream_sdk_<deepstream_version>_jetson.tbz2 -C /
$ cd /opt/nvidia/deepstream/deepstream-5.0
$ sudo ./install.sh
$ sudo ldconfig
```
See [NVIDIA DeepStream SDK Developer Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#install-the-deepstream-sdk) , and more.  


Python Bindings
See [NVIDIA-AI-IOT/deepstream_python_apps/bindings/README.md](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/master/bindings).

```
$ cd /opt/nvidia/deepstream/deepstream-6.0/source
$ sudo git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
$ cd deepstream_python_apps/bindings

$ sudo pip3 install pybind11
$ sudo git submodule update --init
$ sudo mkdir build && cd build
$ sudo cmake ..  -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=6 -DPIP_PLATFORM=linux_aarch64 -DDS_PATH=/opt/nvidia/deepstream/deepstream-6.0/
$ sudo make
```

#### 5. To install TensorRT
JetPack has included TensorRT, only additional packages using Python.  

```
$ sudo apt update
$ sudo apt install tensorrt
$ sudo apt install python3-libnvinfer-dev
```
See [Installation Guide::NVIDIA Deep Learning TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing) , and more.

#### 6. To install torch2trt
```
$ git clone https://github.com/NVIDIA-AI-IOT/torch2trt
$ cd torch2trt
$ sudo python3 setup.py install
```
See [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) , and more.  

#### 7. To install Gstremer
```
$ sudo add-apt-repository universe
$ sudo add-apt-repository multiverse
$ sudo apt update
$ sudo apt install gstreamer1.0-tools gstreamer1.0-alsa \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav
$ sudo apt install libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-good1.0-dev \
  libgstreamer-plugins-bad1.0-dev
```
See [NVIDIA Jetson Linux Developer Guide: Multimedia](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/accelerated_gstreamer.html) , and more.

#### 8. Setup models

```
$ cd <this repo>
$ ./setup.sh
```
Note: It takes about 15 minutes to finish. 

#### 9. Setup Azure, Open Distro for ElasticSearch and Kibana

```
$ pip3 install azure-iot-device
```

Refer to the [README guide](kibana/README.md) in kibana directory for the setup details.
  
 **Note: If you don't need Kibana visualization, skip this step and try running [Without Kibana](#Without-Kibana) .**

### Option 2: Building  docker image 

TBD


## Usage 
### With Kibana
Run the command after starting the Open Distro for Elasticsearch(Elasticsearch, Kibana) service. The data of visualize is sent to Open Distro for Elasticsearch.  

Using video file, path to own video file dir. See `python3 run_gaze_sequential.py -h` for detailed options.
```
$ python3 run_gaze_sequential.py [PATH/TO/VIDEO_DIR/VIDEO] --codec h264
```

Using USB webcam, using `--media` argument, path to USB webcam.  
```
$ pyhon3 run_gaze_sequential.py [PATH/TO/WEBCAM] --media v4l2 
```
### Without Kibana
Only Draw the bounding box, gender and age on the display.

1. Comment out of `run_gaze_sequential.py` the following:
  ```
  #comment out of part1  
    from message_manager import MessageManager
    from config import Config
  
  #comment out of part2 
    config = Config()
    message_manager = MessageManager(config)
    gaze_msg_sender = GazeMessageSender(message_manager, send_msg_interval=5.0)

  #comment out of part3
    gaze_msg_sender(faces, gaze_cpu)
  ``` 

2. Save file and run file  
Same command with Kibana. See With Kibana command.


##  Data analysis and visualization
Data analysis and visualization with Open Distro for Kibana.  
After access and login the Open Distro for Kibana, create Visualization and Dashboard.
Refer to the [README guide](kibana/README.md#Create-visualization-and-Dashboard) in kibana directory for the creating visualization details.

For example, created the Visualization and Dashboard as follow:

![](src/kibana_visualize_v1.jpg)

## Licenses
Copyright (C) 2020, Ryoyo-NV All rights reserved.  
The models are available for research, development, and PoC purposes only.
