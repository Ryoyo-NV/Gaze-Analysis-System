import os
import sys
import ctypes
import cv2
import tensorrt as trt
import numpy as np

sys.path.append("utils")
from trtcuda import TRTForFLM

ctypes.CDLL(os.path.join(os.getcwd(), "libflnetsoftargmax.so"))


class FaceLandmarkEngine(object):

    def __init__(self, path):
        self.module = TRTForFLM(path)
        self.input_dtype = trt.nptype(self.module.engine.get_binding_dtype(0))
        self.output_dtype = trt.nptype(self.module.engine.get_binding_dtype(1))

    def __call__(self, crops):

        num_crops = len(crops)
        crops = crops[:, None, :, :].astype(self.input_dtype)
        landmarks = np.empty((num_crops, 80, 2), dtype=self.output_dtype)

        for i in range(0, num_crops, self.module.max_batch_size):
            lm, _ = self.module([crops[i:i+self.module.max_batch_size]])
            landmarks[i:i+self.module.max_batch_size, :, 0] = lm[:, :80, 0, 0]
            landmarks[i:i+self.module.max_batch_size, :, 1] = lm[:, 80:, 0, 0]

        return landmarks


class FaceLandmarkPost(object):

    def __call__(self, landmarks, crops, boxes):

#        landmarks = landmarks.detach().cpu().numpy()
        crop_width, crop_height = crops.shape[2], crops.shape[1]

        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        landmarks[:, :, 0] = boxes[:, None, 0] + landmarks[:, :, 0] * box_widths[:, None] / crop_width
        landmarks[:, :, 1] = boxes[:, None, 1] + landmarks[:, :, 1] * box_heights[:, None] / crop_height

        return landmarks


def _landmark_minimax_boxes(landmarks, r):
    x0 = np.min(landmarks[:, r, 0], axis=1)
    y0 = np.min(landmarks[:, r, 1], axis=1)
    x1 = np.max(landmarks[:, r, 0], axis=1)
    y1 = np.max(landmarks[:, r, 1], axis=1)
    return np.array([x0, y0, x1, y1]).transpose()


def leye_boxes(landmarks):
    return _landmark_minimax_boxes(landmarks, range(42, 48))


def reye_boxes(landmarks):
    return _landmark_minimax_boxes(landmarks, range(36, 42))


class FaceLandmarkDraw(object):

    def __init__(self, color=(100, 100, 100), radius=2, thickness=-1):
        self.radius = radius
        self.thickness = thickness
        self.color = color

    def __call__(self, image, landmarks):

        for lm in landmarks:
            for xy in lm:
                cv2.circle(image, (int(xy[0]), int(xy[1])), self.radius, self.color, self.thickness)


