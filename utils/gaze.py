import sys
import numpy as np
import json
import datetime


def is_looking(gaze, threshold=100):
    x, y, z, theta, phi = gaze
    return (x**2 + y**2) < threshold**2


class GazeMessageSender:

    def __init__(self, message_manager, send_msg_interval):
        self.message_manager = message_manager
        self.send_msg_interval = send_msg_interval
        self.prev_gazes = {}
        self.prev_send_time = datetime.datetime.now()
        self.records = []

    def __call__(self, faces):

        timestamp = datetime.datetime.now()
        # Check whether the message should be sent at this time or not
        at_msg_send_time = False
        if (timestamp - self.prev_send_time).total_seconds() >= self.send_msg_interval:
            at_msg_send_time = True
            # Update timestamp
            self.prev_send_time = timestamp

        cur_gazes = {}
        for i, face in enumerate(faces):
            face_id = face["id"]
            if len(face["gaze"]) == 0:
                cur_state = False
            else:
                cur_state = is_looking(face["gaze"], threshold=100)
            class GazeState:
                pass
            cur_gaze = GazeState()
            cur_gaze.state = cur_state
            cur_gaze.since = timestamp
            cur_gaze.until = timestamp
            cur_gaze.face = face

            if face_id in self.prev_gazes.keys():
                # Check state diff if face_id exists prev frame already
                prev_gaze = self.prev_gazes.pop(face_id)
                if (prev_gaze.state ^ cur_gaze.state):
                    # Add gaze record if state is changed from previous
                    face["gaze_state"] = bool(prev_gaze.state)
                    face["since"] = prev_gaze.since.isoformat()
                    face["until"] = prev_gaze.until.isoformat()
                    rec = json.dumps(face, ensure_ascii=False)
                    self.records.append(rec)
                else:
                    # Replace since time of current if the state is same as previous
                    cur_gaze.since = prev_gaze.since
                    if at_msg_send_time:
                        # Add record if it is at message sending time
                        face["gaze_state"] = bool(cur_state)
                        face["since"] = cur_gaze.since.isoformat()
                        face["until"] = cur_gaze.until.isoformat()
                        rec = json.dumps(face, ensure_ascii=False)
                        self.records.append(rec)

            cur_gazes[face_id] = cur_gaze

        prev_keys = self.prev_gazes.keys()
        for face_id in prev_keys:
            # Add gaze records remaining in prev_gazes
            gaze = self.prev_gazes[face_id]
            face = gaze.face
            face["gaze_state"] = bool(gaze.state)
            face["since"] = gaze.since.isoformat()
            face["until"] = gaze.until.isoformat()
            rec = json.dumps(face, ensure_ascii=False)
            self.records.append(rec)

        if at_msg_send_time:
            # Change gaze records to a message for sending
            msg = '{"timestamp":"%s",' % timestamp.isoformat()
            msg += '"faces": ['
            for i, rec in enumerate(self.records):
                if i is not 0:
                    msg += ","
                msg += rec
            msg += "]}"
            self.records.clear()

            self.message_manager.add(msg)
            self.message_manager.start()

        # Move current data to previous
        self.prev_gazes = cur_gazes
