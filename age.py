import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt
import torch2trt
import cv2


class AgeEngine(object):

    def __init__(self, path):
        self.device = torch.device('cuda') 

        with open(path, 'rb') as f:
            engine_bytes = f.read()

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.module = torch2trt.TRTModule(
            engine,
            input_names=['input_1'],
            output_names=['output_1']
        ) 
        
        self.max_batch_size = self.module.engine.max_batch_size

    def __call__(self, face_crops):
        
        num_crops = len(face_crops)
        
        predicted_ages = np.empty(0)

        with torch.no_grad():            
            face_crops = torch.from_numpy(np.transpose(face_crops, (0,3,1,2))).to(self.device, dtype=torch.float32)

            for i in range(0, num_crops, self.max_batch_size):
                #predict ages
                outputs = F.softmax(self.module(face_crops[i:i+self.max_batch_size]), dim= -1).cpu().numpy()

                ages = np.arange(0, 101)
                predicted_ages = np.append(predicted_ages, (outputs * ages).sum(axis=-1))
                
        return predicted_ages


class AgeDraw(object):
    def __init__(self, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, color=(255, 255, 255)):
        self.thickness = thickness
        self.color = color
        self.font = font
        self.font_scale = font_scale

    def __call__(self, image, boxes, predicted_ages):
        
        for i, b in enumerate(boxes):

            x0, y0 = int(b[0]), int(b[1])
            x1, y1 = int(b[2]), int(b[3])

            label = "{}".format(int(predicted_ages[i]))
            
            size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0] 
            cv2.rectangle(image, (x0, y0 - size[1]), (x0 + size[0], y0), (100, 100, 100), cv2.FILLED)
            cv2.putText(image, label, (x0, y0), self.font, self.font_scale, self.color, self.thickness, lineType=cv2.LINE_AA)




    

