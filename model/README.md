# Reference of Content Models

Gaze detection system used the content of 5 models.  
Please note that the models are available for research, development, and PoC purposes only.
For uses other than the above, please check the license and replace the model with another model.

Reference of the models link as follows:

- Face detection model baced on NVIDIA Face detection model with input resolution of 384X240 per face. The model was converted from TLT to ETLT, ETLT to TensorRT.  
  - FaceDitectIR : https://ngc.nvidia.com/catalog/models/nvidia:tlt_facedetectir 

- NVIDIA Face landmarks model with input resolution of 80X80 per face. The model was converted from TensorFlow to TensorRT.  
  - Gaze Demo for Jetson: https://ngc.nvidia.com/catalog/containers/nvidia:jetson-gaze  
  - jetson-cloudnative-demo: https://github.com/NVIDIA-AI-IOT/jetson-cloudnative-demo

- NVIDIA Gaze model with input resolution of 224X224 per left eye, right eye and whole face. The model was converted from TensorFlow to TensorRT.  
  - Gaze Demo for Jetson: https://ngc.nvidia.com/catalog/containers/nvidia:jetson-gaze  
  - jetson-cloudnative-demo: https://github.com/NVIDIA-AI-IOT/jetson-cloudnative-demo

- Age estimation model baced on SSR-Net for Age estimation model with input resolution of 64X64 per face. The model was converted form TensorFlow to ONNX, ONNX to TensorRT.  
  - SSR-Net: https://github.com/shamangary/SSR-Net  
  - [MegaAge-Asian dataset](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/)  

- Gender estimation model baced on SSR-Net for Gender estimation model with input resolution of 64X64 per face. The model was converted form TensorFlow to ONNX, ONNX to TensorRT.  
  - SSR-Net: https://github.com/shamangary/SSR-Net  
  - [MegaAge-Asian dataset](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/)  




