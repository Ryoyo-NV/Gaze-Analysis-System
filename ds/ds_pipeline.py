import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

Gst.debug_set_active(True)
Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
GObject.threads_init()
Gst.init(None)

import pyds
import numpy as np
import configparser
import threading
import torch

sys.path.append("ds")

pipeline_playing = False

GIE_ID_DETECT = 1
GIE_ID_AGE = 2
GIE_ID_GENDER = 3

elem_props = {
    "filesrc": {
        "location": ""
    },
    "v4l2src": {
        "device": ""
    },
    "v4l2filter": {
        "caps": "{cap}, width={w}, height={h}, framerate={fps}/1, parsed=True",
    },
    "v4l2jpgdec": {
        "mjpeg": 1
    },
    "filter1": {
        "caps": "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate={fps}/1"
    },
    "streammux": {
        "width": 1280,
        "height": 720,
        "batch-size": 1,
        "batched-push-timeout": 4000000,
        "live-source": 1
    },
    "pgie": {
        "config-file-path": "ds/ds_pgie_config.txt"
    },
    "tracker": {
        "tracker-width": 640,
        "tracker-height": 384,
        "gpu-id": 0,
#        "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_mot_klt.so",
        "ll-lib-file": "/opt/nvidia/deepstream/deepstream-5.0/lib/libnvds_nvdcf.so",
        "ll-config-file": "ds/tracker_config.yml",
        "enable-batch-process": 1
    },
    "sgie1": {
        "config-file-path": "ds/ds_sgie_age_config.txt"
    },
    "sgie2": {
        "config-file-path": "ds/ds_sgie_gender_config.txt"
    },
    "sink": {
#        "emit-signals": True,
        "caps": "video/x-raw, format={form}, width={w}, height={h}, framerate={fps}/1",
        "drop": True,
        "max-buffers": 1,
        "sync": True,
#        "async": True
    }
}

def create_element(pipeline, element, name, props=None):
    new_elem = Gst.ElementFactory.make(element, name)
    if new_elem is not None:
        if props is not None:
            for key, val in props.items():
                if key == "caps":
                    val = Gst.Caps.from_string(val)
                new_elem.set_property(key, val)
        pipeline.add(new_elem)
    else:
        raise ValueError("failed to make %s element '%s'" % (element, name))

    return new_elem


class DsVideo:

    def __init__(self, video_path, media="video", codec="h264", width=1280, height=720, fps=30, format="BGR", is_loopback=False):

        # Create gstreamer pipeline
        #
        self.pipeline = Gst.Pipeline()

        # Set and print properties of elements
        if media == "video":
            elem_props["filesrc"]["location"] = video_path
        elif media == "v4l2":
            elem_props["v4l2src"]["device"] = video_path
            if codec == "mjpg":
                elem_props["v4l2filter"]["caps"] = elem_props["v4l2filter"]["caps"].format(cap="image/jpeg", w=width, h=height, fps=fps)
            else:
                elem_props["v4l2filter"]["caps"] = elem_props["v4l2filter"]["caps"].format(cap="video/x-raw", w=width, h=height, fps=fps)
        elem_props["sink"]["caps"] = elem_props["sink"]["caps"].format(w=width, h=height, form=format, fps=fps)
        elem_props["filter1"]["caps"] = (elem_props["filter1"]["caps"].format(fps=fps))
        print("DeepStream pipeline properties: ", elem_props)

        #
        # Create gstreamer elements
        # source: filesrc or v4l2src
        # decparser: h264parse or h265parse
        # 
        if media == "video":
            source = create_element(self.pipeline, "filesrc", "file-source", elem_props["filesrc"])
            demuxer = create_element(self.pipeline, "qtdemux", "qtdemux")
            queue1 = create_element(self.pipeline, "queue", "demux-queue")
            if codec == "h264":
                decparser = create_element(self.pipeline, "h264parse", "h264parser")
            elif codec == "h265":
                decparser = create_element(self.pipeline, "h265parse", "h265parser")
            else:
                print("Error: Video codec [%s] is not supported" % codec, file=sys.stderr)
            decoder = create_element(self.pipeline, "nvv4l2decoder", "decoder")
        elif media == "v4l2":
            source = create_element(self.pipeline, "v4l2src", "v4l2-source", elem_props["v4l2src"])
            v4l2filter = create_element(self.pipeline, "capsfilter", "v4l2-caps", elem_props["v4l2filter"])
            if codec == "mjpg":
                decoder = create_element(self.pipeline, "nvv4l2decoder", "jpegdecoder", elem_props["v4l2jpgdec"])
            else:
                decoder = create_element(self.pipeline, "videoconvert", "videoconv-for-v4l2")
        else:
            print("Error: Media format [%s] is not supported" % media, file=sys.stderr)
        queue2 = create_element(self.pipeline, "queue", "decode-queue")
        nvvidconv1 = create_element(self.pipeline, "nvvideoconvert", "nvconv-for-streammux")
        filter1 = create_element(self.pipeline, "capsfilter", "nvvidconv1-caps", elem_props["filter1"])
        streammux = create_element(self.pipeline, "nvstreammux", "stream-muxer", elem_props["streammux"])
        pgie = create_element(self.pipeline, "nvinfer", "primary-gie", elem_props["pgie"])
        tracker = create_element(self.pipeline, "nvtracker", "tracker", elem_props["tracker"])
        sgie1 = create_element(self.pipeline, "nvinfer", "sgie-age", elem_props["sgie1"])
        sgie2 = create_element(self.pipeline, "nvinfer", "sgie-gender", elem_props["sgie2"])
        nvvidconv2 = create_element(self.pipeline, "nvvideoconvert", "nvconv-for-sink")
        videoconv1 = create_element(self.pipeline, "videoconvert", "videoconv-for-sink")
        self.sink = create_element(self.pipeline, "appsink", "appsink", elem_props["sink"])

        #
        # Link elements each other
        # For example with h264 video source:
        #   source -> demux -> queue -> h264parse -> decode -> queue -> conv -> caps
        #   -> streammux -> pgie -> tracker -> sgie1 -> sgie2 -> conv -> conv -> sink
        #
        # * demuxer
        # it will be linked with queue1 dynamic because the source pads of
        # demux element are added when extract the mp4 container.
        #
        # * streammux
        # nvstreammux element has a variable number of input streams,
        # so we need to request a sink pad and link with the src pad of
        # previous element(filter1) explicitly.
        #
        if media == "video":
            source.link(demuxer)

            def handle_demux_pad_added(src, new_pad, *args, **kwargs):
                if new_pad.get_name().startswith('video'):
                    queue_sink_pad = queue1.get_static_pad('sink')
                    new_pad.link(queue_sink_pad)
            demuxer.connect("pad-added", handle_demux_pad_added)

            queue1.link(decparser)
            decparser.link(decoder)
        elif media == "v4l2":
            source.link(v4l2filter)
            v4l2filter.link(decoder)

        decoder.link(queue2)
        queue2.link(nvvidconv1)
        nvvidconv1.link(filter1)

        streammux_sinkpad = streammux.get_request_pad("sink_0")
        filter1_srcpad = filter1.get_static_pad("src")
        filter1_srcpad.link(streammux_sinkpad)

        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(sgie1)
        sgie1.link(sgie2)
        sgie2.link(nvvidconv2)
        nvvidconv2.link(videoconv1)

        videoconv1.link(self.sink)

        # Create an event loop and feed gstreamer bus messages to it
        self.loop = GObject.MainLoop()
        self.is_loopback = is_loopback
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", bus_msg_callback_loop, self)

        self.pipeline.set_state(Gst.State.PAUSED)

        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.start()

    def _run(self):
        try:
            self.loop.run()
        except:
            pass
        global pipeline_playing
        pipeline_playing = False

    def start(self):
        # start play back and listen to events
        self.pipeline.set_state(Gst.State.PLAYING)
        global pipeline_playing
        pipeline_playing = True

    def stop(self):
        if self.run_thread.is_alive():
            self.pipeline.set_state(Gst.State.PAUSED)
        global pipeline_playing
        pipeline_playing = False

    def is_alive(self):
        return self.run_thread.is_alive()

    def is_playing(self):
        if not self.run_thread.is_alive():
            return False
        return pipeline_playing

    def quit(self):
        # Shutdown the gstreamer pipeline and thread
        self.pipeline.set_state(Gst.State.PAUSED)
        pyds.unset_callback_funcs()
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()
        self.run_thread.join()
        global pipeline_playing
        pipeline_playing = False

    def read(self):
        image = None
        faces = []

        # Check gstreamer pipeline health
        if not self.run_thread.is_alive():
            print("Error: Failed to read a frame data because gst thread is done already", file=sys.stderr)
            return image, faces
        if not pipeline_playing:
            print("Error: Failed to read a frame data because gst pipeline is stopped", file=sys.stderr)
            return image, faces

        # Pull new buffer from appsink
        sample = self.sink.emit('try-pull-sample', 5000000000)
        if sample is None:
            print("Error: Failed to read a frame data from gst pipeline", file=sys.stderr)
            return image, faces

        # Create ndarray of image from pulled buffer
        new_buffer = sample.get_buffer()
        height = sample.get_caps().get_structure(0).get_value('height')
        width = sample.get_caps().get_structure(0).get_value('width')
        image = np.ndarray(
                        (height, width, 3),
                        buffer=new_buffer.extract_dup(0, new_buffer.get_size()),
                        dtype=np.uint8
                     )

        #
        # Get metadata info from NvDsBatchMeta
        # batch_meta <- frame_meta <- obj_meta <- classfy_meta
        # obj_meta : NvDsObjectMeta metadata from pgie (result of face-detection and tracker)
        # classify_meta : NvDsClassifierMeta metadata from sgie (result of classifications)
        #
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(new_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                face = {}
                rect = {}

                # Pack the result data from DeepStream
                rect["left"] = int(obj_meta.rect_params.left)
                rect["top"] = int(obj_meta.rect_params.top)
                rect["right"] = int(obj_meta.rect_params.left) + int(obj_meta.rect_params.width)
                rect["bottom"] = int(obj_meta.rect_params.top) + int(obj_meta.rect_params.height)
                rect["width"] = int(obj_meta.rect_params.width)
                rect["height"] = int(obj_meta.rect_params.height)
                face["bbox"] = rect

                face["frame"] = int(frame_meta.frame_num)
                face["id"] = int(obj_meta.object_id)

                l_class = obj_meta.classifier_meta_list
                while l_class is not None:
                    try:
                        classify_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
                    except StopIteration:
                        break

                    # Get result of age-estimation
                    if classify_meta.unique_component_id == GIE_ID_AGE:
                        l_label = classify_meta.label_info_list
                        label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                        face["age"] = label_info.result_prob
                    # Get result of gender-estimation
                    if classify_meta.unique_component_id == GIE_ID_GENDER:
                        l_label = classify_meta.label_info_list
                        label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                        face["gender"] = label_info.result_prob

                    try:
                        l_class = l_class.next
                    except StopIteration:
                        break

                faces.append(face)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return image, faces


    def __del__(self):
        self.pipeline.set_state(Gst.State.PAUSED)
        pyds.unset_callback_funcs()
        self.pipeline.set_state(Gst.State.NULL)
        self.loop.quit()
        self.run_thread.join()
        global pipeline_playing
        pipeline_playing = False


# Callback function called from bus message event
def bus_msg_callback_loop(bus, message, dsvideo):
    if message.type == Gst.MessageType.EOS:
        print("Info: End of stream")
        if dsvideo.is_loopback:
            dsvideo.pipeline.seek(1.0, Gst.Format.TIME, Gst.SeekFlags.SEGMENT, Gst.SeekType.SET, 0, Gst.SeekType.NONE, 0)
        else:
            dsvideo.loop.quit()
    elif message.type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print("Warning: %s: %s" % (err, debug), file=sys.stderr)
        dsvideo.loop.quit()
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_warning()
        print("Error: %s: %s\n" % (err, debug), file=sys.stderr)
        dsvideo.loop.quit()
    return True

