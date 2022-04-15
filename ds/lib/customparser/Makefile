## Makefile for age-estimation custom parser in DeepStream

TARGET_LIB:= libcustomparser.so

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CC:= g++

CFLAGS:= -Wall -std=c++11 -shared -fPIC

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes \
	 -I/usr/local/cuda-$(CUDA_VER)/include

LIBS:= -lnvinfer -lnvparsers
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_ageparser.cpp nvdsinfer_genderparser.cpp

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)

clean:
	rm -rf $(TARGET_LIB)
