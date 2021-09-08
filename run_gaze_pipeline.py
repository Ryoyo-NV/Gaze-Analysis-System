import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

import cv2
import time
import argparse
import numpy as np
from box_utils import CropResize, BoxDraw, TextDraw
from face import FaceDetect, FaceDraw
from face_landmark import FaceLandmarkEngine, FaceLandmarkPost, FaceLandmarkDraw, leye_boxes, reye_boxes
from gaze import GazePre, GazeEngine, is_looking
from age import AgeEngine, AgeDraw
from gender import GenderEngine, GenderDraw
import sys

UTILS_PATH = 'utils' 
sys.path.append(UTILS_PATH)

from video import Video
from display import Display
from pipeline import Pipeline, EOS


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--loop', action="store_true")
    parser.add_argument('--codec', type=str, default='h264')
    parser.add_argument('--media', type=str, default='video')
    args = parser.parse_args()

    # media
    video = Video(args.path, args.width, args.height, loop=args.loop, codec=args.codec, media=args.media)
    display = Display(args.width, args.height)

    # face
    face_detect = FaceDetect(shape=(960//2, 540//2))

    # age
    age_crop_resize = CropResize((224, 224), square=True, scale=1.0, channels=True)
    age_engine = AgeEngine('age_model.engine')
    age_draw = AgeDraw()

    # gender estimation
    gender_crop_resize = CropResize((64, 64), square=True, scale=1.0, channels=True)
    gender_engine = GenderEngine('gender_model_wiki.engine')
    gender_draw = GenderDraw()

    # facial landmarks
    flm_crop_resize = CropResize((80, 80), square=True, scale=1.0)
    flm_engine = FaceLandmarkEngine('flm_model.engine')
    flm_post = FaceLandmarkPost()
    flm_draw = FaceLandmarkDraw()

    # gaze
    gaze_eye_crop_resize = CropResize((224, 224), square=True, scale=1.8)
    gaze_face_crop_resize = CropResize((224, 224), square=True, scale=1.3)
    gaze_pre = GazePre((args.width, args.height))
    gaze_engine = GazeEngine('gaze_model.engine')

    box_draw = BoxDraw()
    text_draw = TextDraw()

    t0 = time.time()

    while True:

        image = video.read()
        
        if image is None:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect faces
        face_boxes = face_detect(image)

        # count detected faces 
        n_faces = len(face_boxes)
        print("people: {}".format(n_faces))

        # age estimation
        age_crops, age_boxes = age_crop_resize(rgb, face_boxes)
        ages = age_engine(age_crops)
        #print("age: {}".format(np.round(ages).astype(int)))
        

        # gender estimation
        gender_crops, gender_boxes = gender_crop_resize(rgb, face_boxes)
        genders = gender_engine(gender_crops)
        #print("gender: {}".format(np.round(genders).astype(int))) 
        

        # detect facial landmarks
        flm_crops, flm_boxes = flm_crop_resize(gray, face_boxes)
        flm_lm_gpu_boxframe = flm_engine(flm_crops)
        flm_landmarks = flm_post(flm_lm_gpu_boxframe, flm_crops, flm_boxes)
        #flm_draw(image, flm_landmarks)
        flm_leye_boxes = leye_boxes(flm_landmarks)
        flm_reye_boxes = reye_boxes(flm_landmarks)

        # detect gaze
        gaze_leye_crops, gaze_leye_boxes = gaze_eye_crop_resize(gray, flm_leye_boxes)
        gaze_reye_crops, gaze_reye_boxes = gaze_eye_crop_resize(gray, flm_reye_boxes)
        gaze_face_crops, gaze_face_boxes = gaze_face_crop_resize(gray, face_boxes)
        gaze_leye_crops_gpu, gaze_reye_crops_gpu, gaze_face_crops_gpu, gaze_landmarks_pre_gpu = gaze_pre(gaze_leye_crops, gaze_reye_crops, gaze_face_crops, flm_landmarks)
        gaze_gpu = gaze_engine(gaze_leye_crops_gpu, gaze_reye_crops_gpu, gaze_face_crops_gpu, gaze_landmarks_pre_gpu)
        gaze_cpu = gaze_gpu.detach().cpu().numpy()

        colors = []

        for g in gaze_cpu:
            if is_looking(g, threshold=250):
                colors.append((0, 250, 0))
            else:
                colors.append((100, 100, 100))

        #box_draw(image, face_boxes, (100, 100, 100))
        #box_draw(image, flm_boxes, (100, 100, 100))
        box_draw(image, gaze_leye_boxes, colors)
        box_draw(image, gaze_reye_boxes, colors)
        box_draw(image, gaze_face_boxes, (100, 100, 100))
        #age_draw(image, gaze_face_boxes, ages)
        #gender_draw(image, gaze_face_boxes, genders)
        text_draw(image, gaze_face_boxes, ages, genders)
        

        display.write(image)

        # print FPS
        t1 = time.time()
        print('FPS: %f' % (1.0 / (t1 - t0)))
        t0 = t1
