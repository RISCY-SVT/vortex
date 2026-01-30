ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk
PROJECT := scanout_sim

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc $(VORTEX_HOME)/tests/opencl/video_utils.cc

video_utils.cc.o: $(VORTEX_HOME)/tests/opencl/video_utils.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

OPTS ?= -w 63 -h 65

include ../common.mk
