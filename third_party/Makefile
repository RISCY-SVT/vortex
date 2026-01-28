HOST_CC ?= gcc
HOST_CXX ?= g++

all: cvfpu softfloat ramulator

cvfpu:

softfloat:
	SPECIALIZE_TYPE=RISCV SOFTFLOAT_OPTS="-fPIC -DSOFTFLOAT_ROUND_ODD -DINLINE_LEVEL=5 -DSOFTFLOAT_FAST_DIV32TO16 -DSOFTFLOAT_FAST_DIV64TO32" $(MAKE) -C softfloat/build/Linux-x86_64-GCC

ramulator/libramulator.so:
	cd ramulator && mkdir -p build && cd build && \
		CC=$(HOST_CC) CXX=$(HOST_CXX) cmake .. \
		  -DCMAKE_C_COMPILER=$(HOST_CC) -DCMAKE_CXX_COMPILER=$(HOST_CXX) && \
		make -j4
ramulator: ramulator/libramulator.so

clean:
	$(MAKE) -C softfloat/build/Linux-x86_64-GCC clean
	rm -rf ramulator/build ramulator/libramulator.so

.PHONY: all cvfpu softfloat ramulator
