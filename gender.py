import torch
import tensorrt as trt
import torch2trt
import numpy as np
import cv2

class GenderEngine(object):

    def __init__(self, path):
        self.device = torch.device('cuda')
        self.dtype = torch.float32

        with open(path, 'rb') as f:
            engine_bytes = f.read()

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.module = torch2trt.TRTModule(
            engine,
            input_names=['input_1'],
            output_names=['pred']
        )

        self.max_batch_size = self.module.engine.max_batch_size
        
    def __call__(self, face_crops):
        num_crops = len(face_crops)
        
        genders = np.empty(0, dtype=np.float32)
        face_crops = torch.from_numpy(face_crops).to(self.device, dtype=torch.float32)

        for i in range(0, num_crops, self.max_batch_size):
             preds = self.module(
                face_crops[i:i+self.max_batch_size]
            )
             #genders[i:i+self.max_batch_size] = [n for n in preds]
             #print(preds.detach().cpu())
             genders = np.append(genders, preds.detach().cpu().numpy())

        return genders

class GenderDraw(object):
    def __init__(self, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, color=(255, 255, 255)):
        self.thickness = thickness
        self.color = color
        self.font = font
        self.font_scale = font_scale

    def __call__(self, image, boxes, genders):
        
        for i, b in enumerate(boxes):           

            
            x0, y0 = int(b[0]), int(b[1])
            x1, y1 = int(b[2]), int(b[3])

            if genders[i] >= 0.6:
                label = 'M'
            else:
                label = 'F'

            size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0] 
            cv2.rectangle(image, (x0, y0 - size[1]), (x0 + size[0], y0), (100, 100, 100), cv2.FILLED)
            cv2.putText(image, label, (x0, y0), self.font, self.font_scale, self.color, self.thickness, lineType=cv2.LINE_AA)
