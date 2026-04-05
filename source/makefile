HOST_OS := $(shell uname -s)
ifeq ($(HOST_OS),Darwin)
CXX ?= clang++
else
CXX ?= g++
endif
NVCC ?= nvcc
CUDA ?= 0

BASE_OBJS := Arr_DF.o ApproxBatch.o ApproxBatchCpu.o Coulomb.o ExtMath.o FF.o getparms.o GS.o IDLinterface.o PythonInterface.o Messages.o Plasma.o MWmain.o Std_DF.o Zeta.o
BUILD_INFO := MWTransferArr.buildinfo

ifeq ($(CUDA),1)
CUDA_DEFS := -D MW_USE_CUDA
CUDA_OBJS := ApproxBatchCuda.o
ifneq ($(wildcard /usr/lib/x86_64-linux-gnu/libcudart.so),)
CUDA_RUNTIME_LIBDIR := /usr/lib/x86_64-linux-gnu
else ifneq ($(wildcard /usr/local/cuda/lib64/libcudart.so),)
CUDA_RUNTIME_LIBDIR := /usr/local/cuda/lib64
else ifneq ($(wildcard /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so),)
CUDA_RUNTIME_LIBDIR := /usr/local/cuda/targets/x86_64-linux/lib
else ifneq ($(wildcard /usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudart.so),)
CUDA_RUNTIME_LIBDIR := /usr/local/cuda-12.3/targets/x86_64-linux/lib
else
$(error CUDA=1 requested, but libcudart.so was not found in a supported Linux/WSL location)
endif
CUDA_LDFLAGS := -L$(CUDA_RUNTIME_LIBDIR)
CUDA_LDLIBS := -lcudart
else
CUDA_DEFS :=
CUDA_OBJS := ApproxBatchCudaStub.o
CUDA_LDFLAGS :=
CUDA_LDLIBS :=
endif

ifeq ($(HOST_OS),Darwin)
OMP_CPPFLAGS ?= -I/opt/homebrew/opt/libomp/include
OMP_CXXFLAGS ?= -Xpreprocessor -fopenmp
OMP_LDFLAGS ?= -L/opt/homebrew/opt/libomp/lib
OMP_LDLIBS ?= -lomp
else ifeq ($(HOST_OS),Linux)
OMP_CPPFLAGS ?=
OMP_CXXFLAGS ?= -fopenmp
OMP_LDFLAGS ?=
OMP_LDLIBS ?= -fopenmp
else
$(warning Unrecognized host OS '$(HOST_OS)'; defaulting to Linux-style OpenMP flags)
OMP_CPPFLAGS ?=
OMP_CXXFLAGS ?= -fopenmp
OMP_LDFLAGS ?=
OMP_LDLIBS ?= -fopenmp
endif

CXX_PATH := $(shell command -v $(CXX) 2>/dev/null)
ifeq ($(strip $(CXX_PATH)),)
$(error CXX '$(CXX)' was not found in PATH)
endif

ifeq ($(CUDA),1)
NVCC_PATH := $(shell command -v $(NVCC) 2>/dev/null)
ifeq ($(strip $(NVCC_PATH)),)
$(error CUDA=1 requested, but NVCC '$(NVCC)' was not found in PATH. Install the CUDA toolkit or rebuild with CUDA=0)
endif
else
NVCC_PATH := not-requested
endif

CPPFLAGS += -D LINUX $(CUDA_DEFS) $(OMP_CPPFLAGS)
CXXFLAGS += -std=c++11 -O3 -fPIC $(OMP_CXXFLAGS)
LDFLAGS += -shared $(OMP_LDFLAGS) $(CUDA_LDFLAGS)
LDLIBS += $(OMP_LDLIBS) $(CUDA_LDLIBS)

.PHONY: all clean info

all: MWTransferArr

info:
	@echo "MWTransferArr build configuration"
	@echo "  host_os: $(HOST_OS)"
	@echo "  cxx: $(CXX)"
	@echo "  cxx_path: $(CXX_PATH)"
	@echo "  cppflags: $(CPPFLAGS)"
	@echo "  cxxflags: $(CXXFLAGS)"
	@echo "  ldflags: $(LDFLAGS)"
	@echo "  ldlibs: $(LDLIBS)"
	@echo "  cuda: $(CUDA)"
	@echo "  nvcc: $(NVCC)"
	@echo "  nvcc_path: $(NVCC_PATH)"
	@echo "  cuda_runtime_libdir: $(CUDA_RUNTIME_LIBDIR)"

MWTransferArr: makefile $(BASE_OBJS) $(CUDA_OBJS)
	@echo "Building MWTransferArr.so"
	@echo "  host_os: $(HOST_OS)"
	@echo "  cxx: $(CXX)"
	@echo "  cxx_path: $(CXX_PATH)"
	@echo "  cuda: $(CUDA)"
	@echo "  nvcc_path: $(NVCC_PATH)"
	@echo "  cuda_runtime_libdir: $(CUDA_RUNTIME_LIBDIR)"
	$(CXX) $(filter %.o,$^) $(LDFLAGS) $(LDLIBS) -o MWTransferArr.so
	@{ \
		printf 'host_os=%s\n' "$(HOST_OS)"; \
		printf 'cxx=%s\n' "$(CXX)"; \
		printf 'cxx_path=%s\n' "$(CXX_PATH)"; \
		printf 'cxx_version=%s\n' "$$($(CXX) --version | head -n 1 | tr '\t' ' ')"; \
		printf 'cppflags=%s\n' "$(CPPFLAGS)"; \
		printf 'cxxflags=%s\n' "$(CXXFLAGS)"; \
		printf 'ldflags=%s\n' "$(LDFLAGS)"; \
		printf 'ldlibs=%s\n' "$(LDLIBS)"; \
		printf 'cuda=%s\n' "$(CUDA)"; \
		printf 'nvcc=%s\n' "$(NVCC)"; \
		printf 'nvcc_path=%s\n' "$(NVCC_PATH)"; \
		printf 'cuda_runtime_libdir=%s\n' "$(CUDA_RUNTIME_LIBDIR)"; \
	} > $(BUILD_INFO)

%.o: %.cpp makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

ApproxBatchCuda.o: ApproxBatchCuda.cu ApproxBatch.h ApproxBatchInternal.h Plasma.h ExtMath.h makefile
	$(NVCC) -c -O3 -std=c++14 -Xcompiler -fPIC -D LINUX -D MW_USE_CUDA ApproxBatchCuda.cu -o $@

clean:
	rm -f $(BASE_OBJS) ApproxBatchCuda.o ApproxBatchCudaStub.o MWTransferArr.so $(BUILD_INFO)
