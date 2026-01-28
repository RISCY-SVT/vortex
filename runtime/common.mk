ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

HOST_CC ?= gcc
HOST_CXX ?= g++

CC := $(HOST_CC)
CXX := $(HOST_CXX)

SIM_DIR := $(VORTEX_HOME)/sim
HW_DIR := $(VORTEX_HOME)/hw

INC_DIR := $(VORTEX_HOME)/runtime/include
RT_COMMON_DIR := $(VORTEX_HOME)/runtime/common
