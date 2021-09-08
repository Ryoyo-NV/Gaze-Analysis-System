import numpy as np
from pycuda.autoinit import context
import pycuda.driver as cuda
import tensorrt as trt


class TRTEngine(object):

    def __init__(self, engine_file_path):
        self.path = engine_file_path

        with open(engine_file_path, 'rb') as f:
            engine_bytes = f.read()

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.max_batch_size = self.engine.max_batch_size

        self.inputs = []
        self.outputs = []
        self.bindings = []
        for i, binding in enumerate(self.engine):
            # Allocate CUDA host and device buffers
            shape = (self.max_batch_size, *(self.context.get_binding_shape(i)))
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_buf = cuda.pagelocked_empty(size, dtype)
            dev_buf = cuda.mem_alloc(host_buf.nbytes)
            class CUDAMemory():
                pass
            mem = CUDAMemory()
            mem.host = host_buf
            mem.device = dev_buf
            mem.shape = shape

            # Append buffers to inputs and outputs list
            if self.engine.binding_is_input(i):
                self.inputs.append(mem)
            else:
                self.outputs.append(mem)
            # Append the device buffer to bindings for inference
            self.bindings.append(int(dev_buf))


    def __call__(self, inputs):

        for i, input_buf in enumerate(self.inputs):
            input_buf.host = np.copy(inputs[i].reshape(-1))
        num_inputs = i+1
        batch = (inputs[0].shape)[0]

        [cuda.memcpy_htod_async(mem.device, mem.host, self.stream) for mem in self.inputs]
        self.context.execute_async(batch_size=batch, bindings=self.bindings, stream_handle=self.stream.handle)
        [cuda.memcpy_dtoh_async(mem.host, mem.device, self.stream) for mem in self.outputs]
        self.stream.synchronize()

        outputs = []
        for i, result in enumerate(self.outputs):
            # Reshape the result data to the max-batch shape
            # of numpy and then round down it to the batch size
            data = np.array(result.host)
            data = data.reshape(result.shape)
            data = data[:batch]
            outputs.append(data)

        return outputs


class TRTForFLM(object):
    #
    # The facial landmark model we used returns erroenous values
    # when multi-batch images supplied.
    # No problem at single image(batch=1). This bug is under investigation.
    #
    # We implement this class for workaround against the bug.
    # For now, expand the batched images and infer with single batch in loop with async.
    # 
    def __init__(self, engine_file_path):
        self.path = engine_file_path

        with open(engine_file_path, 'rb') as f:
            engine_bytes = f.read()

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        self.context = self.engine.create_execution_context()
        self.max_batch_size = self.engine.max_batch_size
        self.stream = cuda.Stream()

        self.batched_inputs = []
        self.batched_outputs = []
        self.batched_bindings = []
        for k in range(self.max_batch_size):
            inputs = []
            outputs = []
            bindings = []
            # Allocate CUDA host and device buffers per batch
            for i, binding in enumerate(self.engine):
                shape = (self.context.get_binding_shape(i))
                size = trt.volume(shape)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                host_buf = cuda.pagelocked_empty(size, dtype)
                dev_buf = cuda.mem_alloc(host_buf.nbytes)
                class CUDAMemory():
                    pass
                mem = CUDAMemory()
                mem.host = host_buf
                mem.device = dev_buf
                mem.shape = shape

                # Append buffers to inputs and outputs list
                if self.engine.binding_is_input(i):
                    inputs.append(mem)
                else:
                    outputs.append(mem)
                # Append the device buffer to bindings for inference
                bindings.append(int(dev_buf))

            self.batched_inputs.append(inputs)
            self.batched_outputs.append(outputs)
            self.batched_bindings.append(bindings)


    def __call__(self, inputs):

        batch = (inputs[0].shape)[0]
        for k in range(batch):
            for i, input_buf in enumerate(self.batched_inputs[k]):
                inp = inputs[i][k]
                input_buf.host = np.copy(inp.reshape(-1))

        # Issue tasks per batch with async
        for k in range(batch):
            [cuda.memcpy_htod_async(mem.device, mem.host, self.stream) for mem in self.batched_inputs[k]]
            self.context.execute_async(batch_size=1, bindings=self.batched_bindings[k], stream_handle=self.stream.handle)
            [cuda.memcpy_dtoh_async(mem.host, mem.device, self.stream) for mem in self.batched_outputs[k]]

        self.stream.synchronize()

        outputs = []
        for i in range(len(self.batched_outputs[0])):
            results = np.empty(0)
            for k in range(batch):
                # Stack the result data to the batch
                # Adjust the outputs format for TRTEngine
                data = np.array(self.batched_outputs[k][i].host)
                results = np.append(results, data)
            # Reshape the output buffers to batched tensor
            shape = (batch, *(self.batched_outputs[0][i].shape))
            outputs.append(results.reshape(shape))

        return outputs

