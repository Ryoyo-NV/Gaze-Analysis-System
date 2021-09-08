## Makefile for age-estimation custom parser in DeepStream

CC:= g++

CFLAGS:= -Wall -std=c++11 -shared -fPIC

CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes

LIBS:= -lnvinfer -lnvparsers
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_ageparser.cpp nvdsinfer_genderparser.cpp
TARGET_LIB:= libcustomparser.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

install: $(TARGET_LIB)

clean:
	rm -rf $(TARGET_LIB)
