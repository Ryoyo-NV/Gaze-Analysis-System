#!/usr/bin/env python3

import sys
import time
import argparse
import traceback
import numpy as np

sys.path.append("ds")
from ds.ds_pipeline import DsVideo

sys.path.append("utils")
from display import Display
#from box_utils import BoxDraw, TextDraw
from gaze import is_looking, GazeMessageSender
from message_manager import MessageManager
from config import Config


def main():
        
    try:
        video = None

        parser = argparse.ArgumentParser()
        parser.add_argument('path')
        parser.add_argument('--width', type=int, default=1280)
        parser.add_argument('--height', type=int, default=720)
        parser.add_argument('--codec', choices=['h264', 'h265', 'mjpg'], default='h264')
        parser.add_argument('--media', choices=['video', 'v4l2'], default='video')
        parser.add_argument('--max-fps', type=int, default=30)
        args = parser.parse_args()

        # media
        print("deepstream pipeline initializing...", file=sys.stderr)
        video = DsVideo(args.path, args.media, args.codec, args.width, args.height, args.max_fps, "BGR")

        display = Display(args.width, args.height, format="BGR", fps=args.max_fps)

        config = Config()
        message_manager = MessageManager(config)
        gaze_msg_sender = GazeMessageSender(message_manager, send_msg_interval=10.0)
    
        #box_draw = BoxDraw()
        #text_draw = TextDraw()
    
        video.start()
        t0 = time.time()
    
        while True:

            image, faces = video.read()

            if image is None:
                if not video.is_alive():
                    print("Error: Gst thread is not alive", file=sys.stderr)
                    exit(1)
                if not video.is_playing():
                    print("Warning: Gst pipeline is stopped", file=sys.stderr)
                    video.start()
            else:
    
                n_faces = len(faces)
                if n_faces > 0:

                    face_boxes = np.empty((0, 4), dtype=np.float32)
                    for face in faces:
                        bbox = np.array((face["bbox"]["left"], face["bbox"]["top"],
                                        face["bbox"]["right"], face["bbox"]["bottom"]))
                        face_boxes = np.append(face_boxes, [bbox], axis=0)

                    #print("[DEBUG] people: {}".format(n_faces), file=sys.stderr)

                    # Send gaze data per configured seconds
                    gaze_msg_sender(faces)

                    colors = []

                    for f in faces:
                        g = f["gaze"]
                        if len(g) == 0:
                            colors.append((100, 100, 100))
                            continue
                        if is_looking(g, threshold=250):
                            colors.append((0, 250, 0))
                        else:
                            colors.append((100, 100, 100))

                    try:
                        pass
                        #box_draw(image, face_boxes, colors)
                        #text_draw(image, faces)
                    except (KeyError, ValueError):
                        print("Error: Failed to draw bbox or text", file=sys.stderr)
                        traceback.print_exc()

                display.write(image)

            # print FPS
            dt = time.time() - t0
            lim = 1.0 / args.max_fps
            if dt < lim:
                time.sleep(lim - dt)
            t1 = time.time()
            #print("FPS: %f" % (1.0 / (t1 - t0)), file=sys.stderr)
            t0 = t1

    except Exception as e:
        traceback.print_exc()
        exit(1)
    except (KeyboardInterrupt, SystemExit):
        print("exiting application...", file=sys.stderr)
        exit(1)
    finally:
        if video is not None:
            video.quit()


if __name__ == "__main__":
    sys.exit(main())
