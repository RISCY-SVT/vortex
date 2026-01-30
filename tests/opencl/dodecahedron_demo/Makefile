ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := dodecahedron_demo
SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(VORTEX_HOME)/tests/opencl/video_utils.cc

video_utils.cc.o: $(VORTEX_HOME)/tests/opencl/video_utils.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= --style reference_blue -w 640 -h 640

include $(VORTEX_HOME)/tests/opencl/common.mk
