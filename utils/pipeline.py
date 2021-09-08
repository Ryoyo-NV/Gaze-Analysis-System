import threading
import time
import queue


EOS = -1


class Worker(object):

    def __init__(self, method, in_queue, out_queue):
        self.method = method
        self.in_queue = in_queue
        self.out_queue = out_queue

    def _run(self):
        while True:

            data = self.in_queue.get()
            data = self.method(data)
            self.out_queue.put(data)

            if data is EOS:
                break

    def start(self):
        self.thread = threading.Thread(target=self._run)
        self.thread.start()


class Pipeline(object):

    def __init__(self, methods, queue_size=10):

        self.workers = []

        in_queue = queue.Queue(queue_size)
        out_queue = queue.Queue(queue_size)

        for method in methods:

            self.workers.append(Worker(method, in_queue, out_queue))
            
            in_queue = out_queue
            out_queue = queue.Queue(queue_size)

        self.running = False

    def _put(self):
        while self.running:
            try:
                self.workers[0].in_queue.put_nowait({})
            except:
                time.sleep(1e-6) # 1 microsecond

    def _get(self):
        while True:
            data = self.workers[-1].out_queue.get()
            if data == EOS:
                self.running = False
                break

    def start(self):

        # start workers
        for w in self.workers:
            w.start()

        self.running = True

        # start pusher/puller
        self.put_thread = threading.Thread(target=self._put)
        self.get_thread = threading.Thread(target=self._get)
        self.put_thread.start()
        self.get_thread.start()

    def stop(self):
        self.workers[0].in_queue.put(EOS)
        self.put_thread.join()
        self.get_thread.join()
