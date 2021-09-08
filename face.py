import sys
import cv2
import numpy as np
sys.path.append('tensorrt_demos')
from utils.mtcnn import TrtMtcnn, TrtPNet, TrtRNet, TrtONet


class FaceDetect(TrtMtcnn):

    def __init__(self, shape=(960//4, 540//4)):

        self.pnet = TrtPNet('tensorrt_demos/mtcnn/det1.engine')
        self.rnet = TrtRNet('tensorrt_demos/mtcnn/det2.engine')
        self.onet = TrtONet('tensorrt_demos/mtcnn/det3.engine')
        self.shape = shape
    
    def __call__(self, image):
        width, height = image.shape[1], image.shape[0]
        image = cv2.resize(image, self.shape)
        boxes, landmarks = self.detect(image)
        boxes = np.array(boxes)

        # normalize coordinates
        boxes[:, 0] = width * (boxes[:, 0] + 0.5) / self.shape[0] - 0.5
        boxes[:, 1] = height * (boxes[:, 1] + 0.5) / self.shape[1] - 0.5
        boxes[:, 2] = width * (boxes[:, 2] + 0.5) / self.shape[0] - 0.5
        boxes[:, 3] = height * (boxes[:, 3] + 0.5) / self.shape[1] - 0.5
        
        return boxes


class FaceDraw(object):

    def __init__(self, color=(0, 255, 0), thickness=2):
        self.color = color
        self.thickness = thickness

    def __call__(self, image, boxes):
        for b in boxes:
            x0, y0 = int(b[0]), int(b[1])
            x1, y1 = int(b[2]), int(b[3])

            cv2.rectangle(image, (x0, y0), (x1, y1), self.color, self.thickness)
