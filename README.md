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
- [JetPack](https://developer.nvidia.com/embedded/jetpack) 4.6.1
- Video(H.264/H.265) or USB webcam

Test on:

- Jetson AGX Xavier, JetPack 4.6.1, Video, and USB webcam. 
- Jetson TX2, JetPack 4.6.1

Note: Recommend Jetson Xavier NX or AGX Xavier.  

## Installation

### Requirements:  

- [NVIDIA DeepStream](https://developer.nvidia.com/deepstream-sdk) 6.0.1
- TensorRT 8.2.1
- Gstremer 1.14.5

Optional (using visualize data):
- Open Distro for ElasticSearch 1.10.1
- Open Distro for Kibana 1.10.1
- Azure Iot Hub
- Azure Virtual Machine

Note: Excludes packages already included in JetPack. (ex. Python3.6, OpenCV, and etc.)  
See above for tested versions. 
### Option 1: Installing local  

#### 1. Install DeepStream SDK
```
sudo apt install deepstream-6.0
```
See [NVIDIA DeepStream SDK Developer Guide](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html#install-the-deepstream-sdk) , and more.  


#### 2. Clone this repository & Execute setup.sh
```
git clone https://github.com/Ryoyo-NV/Gaze-Analysis-System
cd Gaze-Analysis-System
chmod +x setup.sh
./setup.sh
```

Note1: setup.sh requires sudo password to install some apt packages.

Note2: It takes about 15 minutes to finish. 


### Option 2: Building docker image 

TBD


## Usage 
Set the path for cvcore_libs to LD_LIBRARY_PATH env.

```
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib/cvcore_libs:$LD_LIBRARY_PATH
```

(Optional) If you use Azure IoT Hub to analize or visualize gaze data, set the connection string in azure_config.ini
```
HOST_NAME = <iot hub hostname>
DEVICE_ID = <iot device id>
SHARED_ACCESS_KEY = <iot device shared access key>
```

With video file, specify the path to your video file. See `python3 run.py -h` for detailed options.
```
python3 run.py [PATH/TO/VIDEO_FILE] --codec h264
```

With USB webcam, specify the path to USB webcam and using `--media` argument.  
```
pyhon3 run.py [PATH/TO/WEBCAM] --media v4l2 
```

Note1: It takes some minutes to run at first time because some models need to convert for TensorRT engines. 

Note2: If you don't use with Azure IoT Hub, you can ignore the message sending errors while running.


##  Data analysis and visualization
Data analysis and visualization with Open Distro for Kibana.  
After access and login the Open Distro for Kibana, create Visualization and Dashboard.
Refer to the [README guide](kibana/README.md#Create-visualization-and-Dashboard) in kibana directory for the creating visualization details.

For example, created the Visualization and Dashboard as follow:

![](src/kibana_visualize_v1.jpg)

## Licenses
Copyright (C) 2020, Ryoyo-NV All rights reserved.  
The models are available for research, development, and PoC purposes only.
