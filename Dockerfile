FROM nvcr.io/nvidia/deepstream-l4t:6.0.1-samples

WORKDIR /root/Gaze-Analysis-System
COPY . .

RUN env

RUN echo -n 'deb https://repo.download.nvidia.com/jetson/common r32.7 main\n\
deb https://repo.download.nvidia.com/jetson/t194 r32.7 main\n'\
>> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list

RUN apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && apt update && apt install -y libopencv-python python3-numpy

RUN bash ./setup.sh
RUN pip3 install Cython --install-option="--no-cython-compile"
RUN echo -n 'export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib/cvcore_libs:$LD_LIBRARY_PATH\n' >> /root/.bashrc
