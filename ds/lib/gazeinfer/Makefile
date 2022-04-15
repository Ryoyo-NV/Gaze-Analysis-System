## Makefile for cpp functions for gazeinfer

APP:= dscprobes.so

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX=g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

##SRCS:= $(wildcard *.c)

INCS:= $(wildcard *.h)

ifeq ($(TARGET_DEVICE),aarch64)
	PKGS:= gstreamer-1.0 python3 pygobject-3.0
else
	PKGS:= gstreamer-1.0 python3 pygobject-3.0
endif

OBJS:= deepstream_faciallandmark_meta.o call_probes_from_py.o

CFLAGS+= -I /opt/nvidia/deepstream/deepstream/sources/includes \
         -I /opt/nvidia/deepstream/deepstream/sources/includes/cvcore_headers \
         -I /usr/local/cuda-$(CUDA_VER)/include \
         -I deepstream_tao_apps/apps/tao_others/deepstream-faciallandmark-app \
         -I deepstream_tao_apps/apps/tao_others/deepstream-gaze-app

CFLAGS+= `pkg-config --cflags $(PKGS)`

CFLAGS+= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-sign-compare -fPIC

LIBS:= `pkg-config --libs $(PKGS)`

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_inferutils \
       -lnvds_utils -lm -lstdc++ \
       -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcuda \
       -L/opt/nvidia/deepstream/deepstream/lib/cvcore_libs \
       -lnvcv_faciallandmarks -lnvcv_core -lnvcv_tensorops -lnvcv_trtbackend \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

all: $(APP)

%.o: %.c $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

deepstream_faciallandmark_meta.o: deepstream_tao_apps/apps/tao_others/deepstream-faciallandmark-app/deepstream_faciallandmark_meta.cpp $(INCS) Makefile
	$(CXX) -c -o $@ -Wall -Werror $(CFLAGS) $<

call_probes_from_py.o: call_probes_from_py.cpp $(INCS) Makefile
	$(CXX) -c -o $@ -fpermissive -Wall -Werror $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -shared -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)
