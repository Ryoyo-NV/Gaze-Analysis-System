import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

Gst.init(None)
Gst.debug_set_active(True)
#Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
GObject.threads_init()

import pyds
import numpy as np
import configparser
import threading
import traceback
import ctypes

sys.path.append("ds")
sys.path.append("ds/lib")
import dscprobes

pipeline_playing = False
pipeline_alive = False
last_sample = None
seek_event = threading.Event()
accumulated_base = 0
prev_accumulated_base = 0

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
        "live-source": False,
        "frame-num-reset-on-eos": True
    },
    "pgie": {
        "config-file-path": "ds/ds_pgie_facedetect_config.txt"
    },
    "tracker": {
        "tracker-width": 640,
        "tracker-height": 384,
        "gpu-id": 0,
        "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file": "ds/tracker_config.yml",
        "enable-batch-process": 1
    },
    "sgie1": {
        "config-file-path": "ds/ds_sgie_age_config.txt"
    },
    "sgie2": {
        "config-file-path": "ds/ds_sgie_gender_config.txt"
    },
    "sgie3": {
        "config-file-path": "ds/ds_sgie_faciallandmarks_config.txt"
    },
    "tgie": {
        "customlib-name": "ds/lib/libnvds_gazeinfer.so",
        "customlib-props": "config-file:ds/lib/gazenet_config.txt"
    },
    "filter2": {
        "caps": "video/x-raw, format={form}, width={w}, height={h}, framerate={fps}/1"
    },
    "filter3": {
        "caps": "video/x-raw, format={form}"
    },
    "sink": {
        "emit-signals": True,
        "caps": "video/x-raw, format={form}, width={w}, height={h}, framerate={fps}/1",
#        "drop": True,
#        "max-buffers": 0,
#        "sync": True,
#        "async": False
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
        elem_props["filter1"]["caps"] = elem_props["filter1"]["caps"].format(fps=fps)
        elem_props["filter2"]["caps"] = elem_props["filter2"]["caps"].format(w=width, h=height, form="RGBA", fps=fps)
        elem_props["filter3"]["caps"] = elem_props["filter3"]["caps"].format(form="BGR")
        elem_props["sink"]["caps"] = elem_props["sink"]["caps"].format(w=width, h=height, form=format, fps=fps)
        print("[DEBUG] DeepStream pipeline properties: ", elem_props)

        #
        # Create gstreamer elements
        # source: filesrc or v4l2src
        # decparser: h264parse or h265parse
        # 
        if media == "video":
            source = create_element(self.pipeline, "uridecodebin", "urisrc", {"uri":"file:///home/ryoyo/gaze_video.mp4"})
            #source = create_element(self.pipeline, "filesrc", "file-source", elem_props["filesrc"])
            #demuxer = create_element(self.pipeline, "qtdemux", "qtdemux")
            queue1 = create_element(self.pipeline, "queue", "demux-queue")
            if codec == "h264":
                decparser = create_element(self.pipeline, "h264parse", "h264parser")
            elif codec == "h265":
                decparser = create_element(self.pipeline, "h265parse", "h265parser")
            else:
                print("Error: Video codec [%s] is not supported" % codec, file=sys.stderr)
            #decoder = create_element(self.pipeline, "nvv4l2decoder", "decoder")
        elif media == "v4l2":
            source = create_element(self.pipeline, "v4l2src", "v4l2-source", elem_props["v4l2src"])
            v4l2filter = create_element(self.pipeline, "capsfilter", "v4l2-caps", elem_props["v4l2filter"])
            if codec == "mjpg":
                decoder = create_element(self.pipeline, "nvv4l2decoder", "jpegdecoder", elem_props["v4l2jpgdec"])
            else:
                decoder = create_element(self.pipeline, "videoconvert", "videoconv-for-v4l2")
        else:
            print("Error: Media format [%s] is not supported" % media, file=sys.stderr)
        #queue2 = create_element(self.pipeline, "queue", "decode-queue")
        nvconv1 = create_element(self.pipeline, "nvvideoconvert", "nvconv-for-streammux")
        filter1 = create_element(self.pipeline, "capsfilter", "caps-for-nvconv1", elem_props["filter1"])
        streammux = create_element(self.pipeline, "nvstreammux", "stream-muxer", elem_props["streammux"])
        pgie = create_element(self.pipeline, "nvinfer", "primary-gie", elem_props["pgie"])
        tracker = create_element(self.pipeline, "nvtracker", "tracker", elem_props["tracker"])
        sgie1 = create_element(self.pipeline, "nvinfer", "sgie-age", elem_props["sgie1"])
        sgie2 = create_element(self.pipeline, "nvinfer", "sgie-gender", elem_props["sgie2"])
        sgie3 = create_element(self.pipeline, "nvinfer", "sgie-fpe", elem_props["sgie3"])
        tgie = create_element(self.pipeline, "nvdsvideotemplate", "tgie-gaze", elem_props["tgie"])
        nvtile = create_element(self.pipeline, "nvmultistreamtiler", "nvtile")
        nvconv2 = create_element(self.pipeline, "nvvideoconvert", "nvconv-for-sink")
        filter2 = create_element(self.pipeline, "capsfilter", "caps-for-nvconv2", elem_props["filter2"])
        vidconv = create_element(self.pipeline, "videoconvert", "videoconv-for-sink")
        filter3 = create_element(self.pipeline, "capsfilter", "caps-for-sink", elem_props["filter3"])
        self.sink = create_element(self.pipeline, "appsink", "appsink", elem_props["sink"])
        self.sink.connect("new-sample", new_sample_callback, None)

        # set pad buffer probe callbacks
        tmp_pad = pgie.get_static_pad('src')
        tmp_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_pad_buffer_probe, None)
        tmp_pad = sgie3.get_static_pad('src')
        tmp_pad.add_probe(Gst.PadProbeType.BUFFER, sgie3_pad_buffer_probe, None)
        tmp_pad = nvtile.get_static_pad('sink')
        tmp_pad.add_probe(Gst.PadProbeType.BUFFER, nvtile_pad_buffer_probe, None)
        if media == "video" and is_loopback == True:
            tmp_pad = queue1.get_static_pad('sink')
            tmp_pad.add_probe(Gst.PadProbeType.EVENT_BOTH | Gst.PadProbeType.EVENT_FLUSH, seek_stream_event, None)
            tmp_pad.add_probe(Gst.PadProbeType.BUFFER, seek_stream_buffer_probe, None)

        # init fpe postprocess for buffer probe
        dscprobes.init_fpe_postprocess(num=80, max_bsize=32, in_width=80, in_height=80)

        #
        # Link elements each other
        # For example with h264 video source:
        #   source -> demux -> queue -> h264parse -> decode -> queue -> conv -> caps
        #   -> streammux -> pgie -> tracker -> sgie1 -> sgie2 -> sgie3 -> tgie
        #   -> conv -> vidconv -> sink
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

            #source.link(demuxer)

            def handle_demux_pad_added(src, new_pad, *args, **kwargs):
                #if new_pad.get_name().startswith('video'):
                if new_pad.get_name().startswith('src'):
                    queue1 = self.pipeline.get_by_name('demux-queue')
                    queue_sink_pad = queue1.get_static_pad('sink')
                    new_pad.link(queue_sink_pad)
            #demuxer.connect("pad-added", handle_demux_pad_added)
            source.connect("pad-added", handle_demux_pad_added)

            queue1.link(nvconv1)#test
            #queue1.link(decparser)
            #decparser.link(decoder)

        elif media == "v4l2":
            source.link(v4l2filter)
            v4l2filter.link(decoder)

        #decoder.link(queue2)
        #queue2.link(nvconv1)
        nvconv1.link(filter1)

        streammux_sinkpad = streammux.get_request_pad("sink_0")
        filter1_srcpad = filter1.get_static_pad("src")
        filter1_srcpad.link(streammux_sinkpad)

        streammux.link(pgie)
        pgie.link(tracker)
        tracker.link(sgie1)
        sgie1.link(sgie2)
        sgie2.link(sgie3)
        sgie3.link(tgie)
        tgie.link(nvtile)
        nvtile.link(nvconv2)
        nvconv2.link(filter2)
        filter2.link(vidconv)
        vidconv.link(filter3)

        filter3.link(self.sink)

        # Create an event loop and feed gstreamer bus messages to it
        self.loop = GObject.MainLoop()
        self.is_loopback = is_loopback
        bus = self.pipeline.get_bus()
        bus.connect("message", bus_msg_callback_loop, self)
        bus.add_signal_watch()

        self.run_thread = threading.Thread(target=self._run, daemon=True)
        self.run_thread.start()

        #self.pipeline.set_state(Gst.State.PAUSED)

        if media == "video" and is_loopback == True:
            #seek_event = threading.Event()
            _, self.duration = self.pipeline.query_duration(Gst.Format.TIME)
            self.seek_thread = threading.Thread(target=self._seek, daemon=True)
            self.seek_thread.start()

    def _run(self):
        try:
            global pipeline_alive
            pipeline_alive = True
            self.loop.run()
        except:
            print("Error: ", sys.exc_info()[0], file=sys.stderr)
        # Shutdown the gstreamer pipeline and thread
        #self.pipeline.set_state(Gst.State.PAUSED)
        print("Info: Gst thread is done", file=sys.stderr)

    def _seek(self):
        offset = 0
        src = self.pipeline.get_by_name('urisrc')
        #queue = self.pipeline.get_by_name('demux-queue')
        #qpad = queue.get_static_pad('sink')
        while True:
            seek_event.wait()

            print("Info: Seeking stream...", file=sys.stderr)
            self.pipeline.set_state(Gst.State.PAUSED)
            src.seek(1.0, Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT, Gst.SeekType.SET, 0, Gst.SeekType.NONE, 0)
            self.pipeline.set_state(Gst.State.PLAYING)
            self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            #offset += self.duration
            #qpad.set_offset(offset)

            seek_event.clear()

    def start(self):
        # start play back and listen to events
        if self.run_thread.is_alive():
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("Error: Failed to set the pipeline state playing")
                return False
            global pipeline_playing
            pipeline_playing = True
        else:
            print("Error: gst pipeline cannot start because gst thread is not alive", file=sys.stderr)

        return True

    def stop(self):
        if self.run_thread.is_alive():
            self.pipeline.set_state(Gst.State.PAUSED)
        global pipeline_playing
        pipeline_playing = False

    def is_alive(self):
        if self.run_thread.is_alive():
            return pipeline_alive
        return False

    def is_playing(self):
        if not self.run_thread.is_alive():
            return False
        return pipeline_playing

    def _quit(self):
        self.pipeline.set_state(Gst.State.NULL)
        global pipeline_playing, pipeline_alive
        pipeline_playing = False
        pipeline_alive = False
        pyds.unset_callback_funcs()
        self.pipeline.unref()
        self.loop.quit()

    def quit(self):
        self.loop.quit()
        if self.run_thread.is_alive():
            self.run_thread.join()
        global pipeline_playing, pipeline_alive
        pipeline_playing = False
        pipeline_alive = False

    def read(self):
        image = None
        faces = []

        # Check gstreamer pipeline health
        if not self.is_alive():
            print("Error: Failed to read a frame data because gst thread is done already", file=sys.stderr)
            return image, faces
        if not self.is_playing():
            print("Error: Failed to read a frame data because gst pipeline is stopped", file=sys.stderr)
            return image, faces

        # Pull new buffer from appsink
        global last_sample
        sample = last_sample
        if (self.sink.props.eos and not(self.is_loopback)):
            print("Error: end of stream", file=sys.stderr)
            raise Exception("end of stream")
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
        if batch_meta is None:
            print("Error: batch meta data is not found in the buffer")
            return image, faces
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
                gaze = {}

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

                l_user = obj_meta.obj_user_meta_list
                while l_user is not None:
                    try:
                        user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                    except StopIteration:
                        break

                    if user_meta.base_meta.meta_type == \
                       pyds.nvds_get_user_meta_type("NVIDIA.RIVA.USER_META_GAZE"):
                        gaze = dscprobes.get_gaze_from_usermeta(user_meta.user_meta_data)

                    try:
                        l_user = l_user.next
                    except StopIteration:
                        break

                face["gaze"] = gaze
                faces.append(face)

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
            except:
                traceback.print_exc()

        return image, faces


    def __del__(self):
        self.quit()


# Callback function called from bus message event
def bus_msg_callback_loop(bus, message, dsvideo):
    global pipeline_alive
    if message.type == Gst.MessageType.ASYNC_DONE:
        pass
    elif message.type == Gst.MessageType.SEGMENT_DONE:
        print("Info: Stream segment done")
        if dsvideo.is_loopback:
            print("Info: Seeking stream...")
            dsvideo.pipeline.set_state(Gst.State.PAUSED)
            dsvideo.pipeline.seek_simple(Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.SEGMENT | Gst.SeekFlags.KEY_UNIT, 0)
            #dsvideo.pipeline.seek(1.0, Gst.Format.TIME, Gst.SeekFlags.FLUSH | Gst.SeekFlags.SEGMENT | Gst.SeekFlags.KEY_UNIT, Gst.SeekType.SET, 0, Gst.SeekType.NONE, 0)
            dsvideo.pipeline.set_state(Gst.State.PLAYING)
            dsvideo.pipeline.get_state(Gst.CLOCK_TIME_NONE)
    elif message.type == Gst.MessageType.EOS:
        print("Info: End of stream")
        if dsvideo.is_loopback:
            pass
        else:
            pipeline_alive = False
            dsvideo._quit()
    elif message.type == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        print("Warning: %s: %s" % (err, debug), file=sys.stderr)
        pipeline_alive = False
        dsvideo._quit()
    elif message.type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print("Error: %s: %s" % (err, debug), file=sys.stderr)
        pipeline_alive = False
        dsvideo._quit()
    return True

# Callback function called from appsink for new-sample event
def new_sample_callback(sink, data):
    global last_sample
    last_sample = sink.emit("pull-sample")
    #print("[DEBUG] Timestamp: ", last_sample.get_buffer().pts)
    return Gst.FlowReturn.OK

# Callback function for pgie(face detection) buffer probe
def pgie_pad_buffer_probe(pad, info, udata):
    dscprobes.facenet_pad_buffer_probe(pad, info, udata)
    return Gst.PadProbeReturn.OK

# Callback function for sgie3(faciallandmark) buffer probe
def sgie3_pad_buffer_probe(pad, info, udata):
    dscprobes.fpenet_pad_buffer_probe(pad, info, udata)
    return Gst.PadProbeReturn.OK

# Callback function for nvtile(nvmultistreamtiler) buffer probe
def nvtile_pad_buffer_probe(pad, info, udata):
    dscprobes.nvtile_pad_buffer_probe(pad, info, udata)
    return Gst.PadProbeReturn.OK

# Callback function for video looping
def seek_stream_event(pad, info, udata):
    event = info.get_event()
    if event is not None:
        if event.type == Gst.EventType.EOS:
            print("[DEBUG] event EOS", file=sys.stderr)
            seek_event.set()
        if event.type == Gst.EventType.SEGMENT:
            print("[DEBUG] event SEGMENT", file=sys.stderr)
            global accumulated_base, prev_accumulated_base
            seg = Gst.Segment.new()
            event.copy_segment(seg)
            seg.base = accumulated_base
            prev_accumulated_base = accumulated_base
            accumulated_base = seg.stop
            event.new_segment(seg)
        if event.type == Gst.EventType.SEGMENT_DONE:
            print("[DEBUG] event SEGMENT_DONE", file=sys.stderr)
        if event.type == Gst.EventType.FLUSH_START or event.type == Gst.EventType.FLUSH_STOP \
                or event.type == Gst.EventType.EOS or event.type == Gst.EventType.QOS or event.type == Gst.EventType.SEGMENT:
            print("[DEBUG] event dropped", file=sys.stderr)
            return Gst.PadProbeReturn.DROP
    return Gst.PadProbeReturn.OK

def seek_stream_buffer_probe(pad, info, udata):
    buf = info.get_buffer()
    buf.pts += prev_accumulated_base
    return Gst.PadProbeReturn.OK
