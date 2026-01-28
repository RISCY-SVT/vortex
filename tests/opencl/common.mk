ROOT_DIR := $(realpath ../../..)

HOST_CC ?= gcc
HOST_CXX ?= g++

CC := $(HOST_CC)
CXX := $(HOST_CXX)

OPENCL_DIAG ?= 1
# OPENCL_DIAG_LEVEL: brief|full (or 1|2). brief prints summary, full prints limits.
OPENCL_DIAG_LEVEL ?= brief
export OPENCL_DIAG_LEVEL
OPENCL_DIAG_TOOL := $(ROOT_DIR)/tests/opencl/ocl_diag
OPENCL_DIAG_SRC := $(firstword \
	$(wildcard $(ROOT_DIR)/tests/opencl/ocl_diag.cc) \
	$(wildcard $(ROOT_DIR)/../tests/opencl/ocl_diag.cc))
OPENCL_DIAG_HDR := $(firstword \
	$(wildcard $(ROOT_DIR)/tests/opencl/ocl_diag.h) \
	$(wildcard $(ROOT_DIR)/../tests/opencl/ocl_diag.h))
OPENCL_DIAG_SCRIPT := $(firstword \
	$(wildcard $(ROOT_DIR)/tests/opencl/run_with_diag.sh) \
	$(wildcard $(ROOT_DIR)/../tests/opencl/run_with_diag.sh))

TARGET ?= opaesim

XRT_SYN_DIR ?= $(VORTEX_HOME)/hw/syn/xilinx/xrt
XRT_DEVICE_INDEX ?= 0

STARTUP_ADDR ?= 0x80000000

ifeq ($(XLEN),64)
	ifeq ($(EXT_V_ENABLE),1)
		VX_CFLAGS += -march=rv64imafdv_zve64d -mabi=lp64d # vector extension
	else
		VX_CFLAGS += -march=rv64imafd -mabi=lp64d
	endif
	POCL_CC_FLAGS += POCL_VORTEX_XLEN=64
else
	ifeq ($(EXT_V_ENABLE),1)
		VX_CFLAGS += -march=rv32imafv_zve32f -mabi=ilp32f # vector extension
	else
		VX_CFLAGS += -march=rv32imaf -mabi=ilp32f
	endif
	POCL_CC_FLAGS += POCL_VORTEX_XLEN=32
endif

VORTEX_RT_PATH ?= $(ROOT_DIR)/runtime
VORTEX_KN_PATH ?= $(ROOT_DIR)/kernel

POCL_PATH ?= $(TOOLDIR)/pocl

LLVM_POCL ?= $(TOOLDIR)/llvm-vortex

VX_LIBS += -L$(LIBC_VORTEX)/lib -lm -lc

VX_LIBS += $(LIBCRT_VORTEX)/lib/baremetal/libclang_rt.builtins-riscv$(XLEN).a
#VX_LIBS += -lgcc

VX_CFLAGS  += -O3 -mcmodel=medany --sysroot=$(RISCV_SYSROOT) --gcc-toolchain=$(RISCV_TOOLCHAIN_PATH)
VX_CFLAGS  += -fno-rtti -fno-exceptions -nostartfiles -nostdlib -fdata-sections -ffunction-sections
VX_CFLAGS  += -I$(ROOT_DIR)/hw -I$(VORTEX_HOME)/kernel/include -DXLEN_$(XLEN) -DNDEBUG $(CONFIGS)
VX_CFLAGS  += -Xclang -target-feature -Xclang +vortex
VX_CFLAGS  += -Xclang -target-feature -Xclang +zicond
VX_CFLAGS  += -mllvm -disable-loop-idiom-all	# disable memset/memcpy loop replacement
#VX_CFLAGS += -mllvm -vortex-branch-divergence=0
#VX_CFLAGS += -mllvm -debug -mllvm -print-after-all

VX_LDFLAGS += -Wl,-Bstatic,--gc-sections,-T$(VORTEX_HOME)/kernel/scripts/link$(XLEN).ld,--defsym=STARTUP_ADDR=$(STARTUP_ADDR) $(VORTEX_KN_PATH)/libvortex.a $(VX_LIBS)

VX_BINTOOL += OBJCOPY=$(LLVM_VORTEX)/bin/llvm-objcopy $(VORTEX_HOME)/kernel/scripts/vxbin.py

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -Wno-deprecated-declarations -Wno-unused-parameter -Wno-narrowing
CXXFLAGS += -pthread
CXXFLAGS += -I$(POCL_PATH)/include
CXXFLAGS += $(CONFIGS)

POCL_CC_FLAGS += LLVM_PREFIX=$(LLVM_VORTEX) POCL_VORTEX_BINTOOL="$(VX_BINTOOL)" POCL_VORTEX_CFLAGS="$(VX_CFLAGS)" POCL_VORTEX_LDFLAGS="$(VX_LDFLAGS)"

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0
	POCL_CC_FLAGS += POCL_DEBUG=all
else
	CXXFLAGS += -O2 -DNDEBUG
endif

LDFLAGS += -Wl,-rpath,$(LLVM_VORTEX)/lib

ifeq ($(TARGET), fpga)
	OPAE_DRV_PATHS ?= libopae-c.so
else
ifeq ($(TARGET), asesim)
	OPAE_DRV_PATHS ?= libopae-c-ase.so
else
ifeq ($(TARGET), opaesim)
	OPAE_DRV_PATHS ?= libopae-c-sim.so
endif
endif
endif

OBJS := $(addsuffix .o, $(notdir $(SRCS)))

all: $(PROJECT)

%.cc.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.cpp.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.c.o: $(SRC_DIR)/%.c
	$(CC) $(CXXFLAGS) -c $< -o $@

$(PROJECT): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -L$(VORTEX_RT_PATH) -lvortex -L$(POCL_PATH)/lib -lOpenCL -o $@

$(PROJECT).host: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -lOpenCL -o $@

$(OPENCL_DIAG_TOOL): $(OPENCL_DIAG_SRC) $(OPENCL_DIAG_HDR)
	$(CXX) $(CXXFLAGS) $(OPENCL_DIAG_SRC) $(LDFLAGS) -L$(LLVM_VORTEX)/lib -L$(POCL_PATH)/lib -lOpenCL -o $@

run-gpu: $(PROJECT).host $(KERNEL_SRCS)
ifeq ($(OPENCL_DIAG),1)
	$(OPENCL_DIAG_SCRIPT) --tool $(OPENCL_DIAG_TOOL) --project $(PROJECT).host -- ./$(PROJECT).host $(OPTS)
else
	./$(PROJECT).host $(OPTS)
endif

run-simx: $(PROJECT) $(KERNEL_SRCS) $(OPENCL_DIAG_TOOL)
ifeq ($(OPENCL_DIAG),1)
	LD_LIBRARY_PATH=$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=simx $(OPENCL_DIAG_SCRIPT) --tool $(OPENCL_DIAG_TOOL) --project $(PROJECT) -- ./$(PROJECT) $(OPTS)
else
	LD_LIBRARY_PATH=$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=simx ./$(PROJECT) $(OPTS)
endif

run-rtlsim: $(PROJECT) $(KERNEL_SRCS) $(OPENCL_DIAG_TOOL)
ifeq ($(OPENCL_DIAG),1)
	LD_LIBRARY_PATH=$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=rtlsim $(OPENCL_DIAG_SCRIPT) --tool $(OPENCL_DIAG_TOOL) --project $(PROJECT) -- ./$(PROJECT) $(OPTS)
else
	LD_LIBRARY_PATH=$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=rtlsim ./$(PROJECT) $(OPTS)
endif

run-opae: $(PROJECT) $(KERNEL_SRCS) $(OPENCL_DIAG_TOOL)
ifeq ($(OPENCL_DIAG),1)
	SCOPE_JSON_PATH=$(VORTEX_RT_PATH)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=opae $(OPENCL_DIAG_SCRIPT) --tool $(OPENCL_DIAG_TOOL) --project $(PROJECT) -- ./$(PROJECT) $(OPTS)
else
	SCOPE_JSON_PATH=$(VORTEX_RT_PATH)/scope.json OPAE_DRV_PATHS=$(OPAE_DRV_PATHS) LD_LIBRARY_PATH=$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=opae ./$(PROJECT) $(OPTS)
endif

run-xrt: $(PROJECT) $(KERNEL_SRCS) $(OPENCL_DIAG_TOOL)
ifeq ($(TARGET), hw)
ifeq ($(OPENCL_DIAG),1)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(VORTEX_RT_PATH)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=xrt $(OPENCL_DIAG_SCRIPT) --tool $(OPENCL_DIAG_TOOL) --project $(PROJECT) -- ./$(PROJECT) $(OPTS)
else
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XRT_INI_PATH=$(VORTEX_RT_PATH)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
endif
else ifeq ($(TARGET), hw_emu)
ifeq ($(OPENCL_DIAG),1)
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(VORTEX_RT_PATH)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=xrt $(OPENCL_DIAG_SCRIPT) --tool $(OPENCL_DIAG_TOOL) --project $(PROJECT) -- ./$(PROJECT) $(OPTS)
else
	SCOPE_JSON_PATH=$(FPGA_BIN_DIR)/scope.json XCL_EMULATION_MODE=$(TARGET) XRT_INI_PATH=$(VORTEX_RT_PATH)/xrt/xrt.ini EMCONFIG_PATH=$(FPGA_BIN_DIR) XRT_DEVICE_INDEX=$(XRT_DEVICE_INDEX) XRT_XCLBIN_PATH=$(FPGA_BIN_DIR)/vortex_afu.xclbin LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
endif
else
ifeq ($(OPENCL_DIAG),1)
	SCOPE_JSON_PATH=$(VORTEX_RT_PATH)/scope.json LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=xrt $(OPENCL_DIAG_SCRIPT) --tool $(OPENCL_DIAG_TOOL) --project $(PROJECT) -- ./$(PROJECT) $(OPTS)
else
	SCOPE_JSON_PATH=$(VORTEX_RT_PATH)/scope.json LD_LIBRARY_PATH=$(XILINX_XRT)/lib:$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=xrt ./$(PROJECT) $(OPTS)
endif
endif

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean-kernel:
	rm -rf *.dump *.ll

clean-host:
	rm -rf $(PROJECT) $(PROJECT).host *.o *.log .depend

clean: clean-kernel clean-host

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif
