
ROOT_DIR := $(realpath ../../..)

HOST_CC ?= gcc
HOST_CXX ?= g++

CC := $(HOST_CC)
CXX := $(HOST_CXX)

CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic -Wfatal-errors
CXXFLAGS += -I$(SW_COMMON_DIR)
CXXFLAGS += $(CONFIGS)

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
else
	CXXFLAGS += -O2 -DNDEBUG
endif

all: $(PROJECT)

$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run:
	./$(PROJECT)

clean:
	rm -rf $(PROJECT) *.o *.log .depend

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
