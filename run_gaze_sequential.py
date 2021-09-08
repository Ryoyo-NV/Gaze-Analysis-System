#!/usr/bin/env python3

import cv2
import sys
import time
import signal
import argparse
import traceback
import numpy as np
from box_utils import CropResize, BoxDraw, TextDraw
from face_landmark import FaceLandmarkEngine, FaceLandmarkPost, FaceLandmarkDraw, leye_boxes, reye_boxes
from gaze import GazePre, GazeEngine, is_looking, GazeMessageSender
from age import AgeDraw
from gender import GenderDraw
from message_manager import MessageManager
from config import Config

sys.path.append("ds")
from ds.ds_pipeline import DsVideo

sys.path.append("utils")
from display import Display


def main():
        
    try:
        video = None

        parser = argparse.ArgumentParser()
        parser.add_argument('path')
        parser.add_argument('--width', type=int, default=1280)
        parser.add_argument('--height', type=int, default=720)
        parser.add_argument('--loop', action="store_true")
        parser.add_argument('--codec', type=str, default='h264')
        parser.add_argument('--media', type=str, default='video')
        parser.add_argument('--max-fps', type=int, default=30)
        args = parser.parse_args()

        # media
        print("deepstream pipeline initializing...", file=sys.stderr)
        video = DsVideo(args.path, args.media, args.codec, args.width, args.height, args.max_fps, "BGR", args.loop)
        video.start()

    #    video = Video(args.path, args.width, args.height, loop=args.loop, codec=args.codec, media=args.media)
        display = Display(args.width, args.height, fps=args.max_fps)

        age_draw = AgeDraw()
        gender_draw = GenderDraw()
        text_draw = TextDraw()

    
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

        config = Config()
        message_manager = MessageManager(config)
        gaze_msg_sender = GazeMessageSender(message_manager, send_msg_interval=5.0)
    
        box_draw = BoxDraw()
    
        t0 = time.time()
    
        while True:

            image, faces = video.read()

            if image is None:
                if not video.is_alive():
                    sys.stderr.write("Error: Gst thread is dead \n")
                    exit(1)
            else:
    
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                n_faces = len(faces)
                if n_faces > 0:

                    #print(faces, file=sys.stderr)
                    face_boxes = np.empty((0, 4), dtype=np.float32)
                    ids = np.empty(0, dtype=np.int32)
                    ages = np.empty(0)
                    genders = np.empty(0)
                    for face in faces:
                        bbox = np.array((face["bbox"]["left"], face["bbox"]["top"],
                                        face["bbox"]["right"], face["bbox"]["bottom"]))
                        face_boxes = np.append(face_boxes, [bbox], axis=0)
                        ids = np.append(ids, face["id"])
                        ages = np.append(ages, face["age"])
                        genders = np.append(genders, face["gender"])

                    print("people: {}".format(n_faces), file=sys.stderr)

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
#                    gaze_gpu = gaze_engine(gaze_leye_crops_gpu, gaze_reye_crops_gpu, gaze_face_crops_gpu, gaze_landmarks_pre_gpu)
#                    gaze_cpu = gaze_gpu.detach().cpu().numpy()
                    gaze_cpu = gaze_engine(gaze_leye_crops_gpu, gaze_reye_crops_gpu, gaze_face_crops_gpu, gaze_landmarks_pre_gpu)

                    # Send gaze data per configured seconds
                    gaze_msg_sender(faces, gaze_cpu)

                    colors = []

                    for g in gaze_cpu:
                        if is_looking(g, threshold=250):
                            colors.append((0, 250, 0))
                        else:
                            colors.append((100, 100, 100))
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
            dt = time.time() - t0
            lim = 1.0 / args.max_fps
            if dt < lim:
                time.sleep(lim - dt)
            t1 = time.time()
            print("FPS: %f" % (1.0 / (t1 - t0)), file=sys.stderr)
            t0 = t1

    except Exception as e:
        traceback.print_exc()
        exit(1)
    finally:
        if video is not None:
            video.quit()


def sigterm_handler(signum, data):
    print("call sigterm_handler", file=sys.stderr)
    exit(1)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, sigterm_handler)
    sys.exit(main())