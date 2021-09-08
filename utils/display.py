import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


Gst.init(None)


class Display(object):

    def __init__(self, width, height, max_queue_size=15, fps=30, format="BGR"):
        self.width = width
        self.height = height
        self.fps = fps
        self.format = format
        self.max_queue_size = max_queue_size
        self.reset()

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)

    def reset(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.set_state(Gst.State.NULL)

        self.pipeline = Gst.parse_launch(self._gst_str())
        self.appsrc = self.pipeline.get_by_name('appsrc')
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("block", True)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _gst_str(self):
        return 'appsrc name=appsrc emit-signals=True is-live=True caps=video/x-raw,format={format},width={width},height={height},framerate={fps}/1 ! videoconvert ! queue max-size-buffers={max_queue_size} ! nveglglessink async=1 sync=0 max-lateness=-1 window-width={width} window-height={height} show-preroll-frame=0'.format(width=self.width, height=self.height, max_queue_size=self.max_queue_size, fps=self.fps, format=self.format)

    def write(self, image):
        buffer = Gst.Buffer.new_wrapped(image.tobytes())
        self.appsrc.emit('push-buffer', buffer)

    def destroy(self):
        self.stop()
