ROOT_DIR := $(realpath ../..)
include $(ROOT_DIR)/config.mk

HOST_CC ?= gcc
HOST_CXX ?= g++

CC := $(HOST_CC)
CXX := $(HOST_CXX)

HW_DIR := $(VORTEX_HOME)/hw
RTL_DIR := $(HW_DIR)/rtl
DPI_DIR := $(HW_DIR)/dpi
SCRIPT_DIR := $(HW_DIR)/scripts
