import cv2


class Video(object):

    def __init__(self, path, width, height, qsize=10, loop=True, codec='h264', media='video'):
        self.path = path
        self.width = width
        self.height = height
        self.qsize = qsize
        self.loop = loop
        self.codec = codec
        self.media = media
        self.reset()

    def _gst_str(self):
        pipeline = ''
        if self.media == 'video':
            return 'filesrc location={path} ! qtdemux ! queue max-size-buffers={qsize} ! {codec}parse ! omx{codec}dec ! nvvidconv ! video/x-raw,format=BGRx ! queue max-size-buffers={qsize} ! videoconvert ! queue max-size-buffers={qsize} ! video/x-raw,format=BGR,width={width},height={height} ! appsink sync=0'.format(path=self.path, width=self.width, height=self.height, qsize=self.qsize, codec=self.codec)
        elif self.media == 'v4l2':
            return 'v4l2src device={path} ! queue max-size-buffers={qsize} ! videoconvert ! queue max-size-buffers={qsize} ! video/x-raw,format=BGR,width={width},height={height} ! appsink sync=0'.format(path=self.path, width=self.width, height=self.height, qsize=self.qsize)

    def reset(self):
        if hasattr(self, 'cap'):
            del self.cap
        self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)

    def read(self):
        re, img = self.cap.read()
        if re:
            return img
        elif self.loop:
            self.reset()
            re, img = self.cap.read()
            if re:
                return img
            else:
                return None
        else:
            return None

    def destroy(self):
        self.cap.release()
