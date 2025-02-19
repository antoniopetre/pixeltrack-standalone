export BASE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Build flags
export CXX := g++
USER_CXXFLAGS :=
HOST_CXXFLAGS := -O2 -fPIC -fdiagnostics-show-option -felide-constructors -fmessage-length=0 -fno-math-errno -ftree-vectorize -fvisibility-inlines-hidden --param vect-max-version-for-alias-checks=50 -msse3 -pipe -pthread -Werror=address -Wall -Werror=array-bounds -Wno-attributes -Werror=conversion-null -Werror=delete-non-virtual-dtor -Wno-deprecated -Werror=format-contains-nul -Werror=format -Wno-long-long -Werror=main -Werror=missing-braces -Werror=narrowing -Wno-non-template-friend -Wnon-virtual-dtor -Werror=overflow -Werror=overlength-strings -Wparentheses -Werror=pointer-arith -Wno-psabi -Werror=reorder -Werror=return-local-addr -Wreturn-type -Werror=return-type -Werror=sign-compare -Werror=strict-aliasing -Wstrict-overflow -Werror=switch -Werror=type-limits -Wunused -Werror=unused-but-set-variable -Wno-unused-local-typedefs -Werror=unused-value -Wno-error=unused-variable -Wno-vla -Werror=write-strings
export CXXFLAGS := -std=c++17 $(HOST_CXXFLAGS) $(USER_CXXFLAGS) -g
export LDFLAGS := -O2 -fPIC -pthread -Wl,-E -lstdc++fs -ldl
export LDFLAGS_NVCC := -ccbin $(CXX) --linker-options '-E' --linker-options '-lstdc++fs'
export SO_LDFLAGS := -Wl,-z,defs
export SO_LDFLAGS_NVCC := --linker-options '-z,defs'

GCC_TOOLCHAIN := $(abspath $(dir $(shell which $(CXX)))/..)

CLANG_FORMAT := clang-format-10
CMAKE := cmake

# Source code
export SRC_DIR := $(BASE_DIR)/src

# Directory where to put object and dependency files
export OBJ_DIR := $(BASE_DIR)/obj

# Directory where to put libraries
export LIB_DIR := $(BASE_DIR)/lib

# Directory where to put unit test executables
export TEST_DIR := $(BASE_DIR)/test

# System external definitions
# CUDA
CUDA_BASE := /usr/local/cuda
ifeq ($(wildcard $(CUDA_BASE)),)
# CUDA platform not found
CUDA_BASE :=
else
# CUDA platform at $(CUDA_BASE)
CUDA_LIBDIR := $(CUDA_BASE)/lib64
USER_CUDAFLAGS :=
export CUDA_BASE
export CUDA_DEPS := $(CUDA_LIBDIR)/libcudart.so
export CUDA_ARCH := 35 50 60 70
export CUDA_CXXFLAGS := -I$(CUDA_BASE)/include
export CUDA_TEST_CXXFLAGS := -DGPU_DEBUG
export CUDA_LDFLAGS := -L$(CUDA_LIBDIR) -lcudart -lcudadevrt
export CUDA_NVCC := $(CUDA_BASE)/bin/nvcc
define CUFLAGS_template
$(2)NVCC_FLAGS := $$(foreach ARCH,$(1),-gencode arch=compute_$$(ARCH),code=[sm_$$(ARCH),compute_$$(ARCH)]) -Wno-deprecated-gpu-targets -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info --source-in-ptx --display-error-number --threads $$(words $(1)) --cudart=shared
$(2)NVCC_COMMON := -std=c++17 -O3 -g $$($(2)NVCC_FLAGS) -ccbin $(CXX) --compiler-options '$(HOST_CXXFLAGS) $(USER_CXXFLAGS)'
$(2)CUDA_CUFLAGS := -dc $$($(2)NVCC_COMMON) $(USER_CUDAFLAGS)
$(2)CUDA_DLINKFLAGS := -dlink $$($(2)NVCC_COMMON)
endef
$(eval $(call CUFLAGS_template,$(CUDA_ARCH),))
export CUDA_CUFLAGS
export CUDA_DLINKFLAGS
endif

# ROCm
ROCM_BASE := /usr/local/rocm-4.0.0
ifeq ($(wildcard $(ROCM_BASE)),)
# ROCm platform not found
ROCM_BASE :=
else
# ROCm platform at $(ROCM_BASE)
export HIP_DEPS := $(ROCM_BASE)/lib/libamdhip64.so
export ROCM_HIPCC := $(ROCM_BASE)/bin/hipcc
HIPCC_UNSUPPORTED_CXXFLAGS := --param vect-max-version-for-alias-checks=50 -Werror=format-contains-nul -Wno-non-template-friend -Werror=return-local-addr -Werror=unused-but-set-variable
export HIPCC_CXXFLAGS := -fno-gpu-rdc --amdgpu-target=gfx900 $(filter-out $(HIPCC_UNSUPPORTED_CXXFLAGS),$(CXXFLAGS)) --gcc-toolchain=$(GCC_TOOLCHAIN)
export HIPCC_LDFLAGS := $(LDFLAGS) --gcc-toolchain=$(GCC_TOOLCHAIN)
# flags to be used by GCC when compiling host code that includes hip_runtime.h
HIPCONFIG := $(ROCM_BASE)/bin/hipconfig
export HIP_CXXFLAGS:= $(shell $(HIPCONFIG) --cpp_config)
export HIP_TEST_CXXFLAGS := -DGPU_DEBUG
endif

# Input data definitions
DATA_BASE := $(BASE_DIR)/data
export DATA_DEPS := $(DATA_BASE)/data_ok
DATA_TAR_GZ := $(DATA_BASE)/data.tar.gz

# External definitions
EXTERNAL_BASE := $(BASE_DIR)/external

TBB_BASE := $(EXTERNAL_BASE)/tbb
TBB_LIBDIR := $(TBB_BASE)/lib
TBB_LIB := $(TBB_LIBDIR)/libtbb.so
export TBB_DEPS := $(TBB_LIB)
export TBB_CXXFLAGS := -I$(TBB_BASE)/include
export TBB_LDFLAGS := -L$(TBB_LIBDIR) -ltbb
export TBB_NVCC_CXXFLAGS :=

EIGEN_BASE := $(EXTERNAL_BASE)/eigen
export EIGEN_DEPS := $(EIGEN_BASE)
export EIGEN_CXXFLAGS := -I$(EIGEN_BASE) -DEIGEN_DONT_PARALLELIZE
export EIGEN_LDFLAGS :=
export EIGEN_NVCC_CXXFLAGS := --diag-suppress 20014

BOOST_BASE := /usr
# Minimum required version of Boost, e.g. 1.65.1
BOOST_MIN_VERSION := 106501
# Check if an external version of Boost is present and recent enough
ifeq ($(wildcard $(BOOST_BASE)/include/boost/version.hpp),)
NEED_BOOST := true
else
NEED_BOOST := $(shell awk '/\#define BOOST_VERSION\>/ { if ($$3 < $(BOOST_MIN_VERSION)) print "true" }' $(BOOST_BASE)/include/boost/version.hpp )
endif
ifeq ($(NEED_BOOST),true)
BOOST_BASE := $(EXTERNAL_BASE)/boost
endif
export BOOST_DEPS := $(BOOST_BASE)
export BOOST_CXXFLAGS := -I$(BOOST_BASE)/include
export BOOST_LDFLAGS := -L$(BOOST_BASE)/lib
export BOOST_NVCC_CXXFLAGS :=

BACKTRACE_BASE := $(EXTERNAL_BASE)/libbacktrace
export BACKTRACE_DEPS := $(BACKTRACE_BASE)
export BACKTRACE_CXXFLAGS := -I$(BACKTRACE_BASE)/include
export BACKTRACE_LDFLAGS := -L$(BACKTRACE_BASE)/lib -lbacktrace

HWLOC_BASE := $(EXTERNAL_BASE)/hwloc
export HWLOC_DEPS := $(HWLOC_BASE)
HWLOC_CXXFLAGS := -I$(HWLOC_BASE)/include
HWLOC_LDFLAGS := -L$(HWLOC_BASE)/lib -lhwloc

ALPAKA_BASE := $(EXTERNAL_BASE)/alpaka
export ALPAKA_DEPS := $(ALPAKA_BASE)
export ALPAKA_CXXFLAGS := -I$(ALPAKA_BASE)/include
# Temporarily filter out missing-braces warning, see https://github.com/cms-patatrack/pixeltrack-standalone/issues/126
export ALPAKA_CUFLAGS := $(filter-out -Werror=missing-braces,$(CUDA_CUFLAGS))

CUPLA_BASE := $(EXTERNAL_BASE)/cupla
export CUPLA_DEPS := $(CUPLA_BASE)/lib
export CUPLA_LIBDIR := $(CUPLA_BASE)/lib
export CUPLA_CXXFLAGS := -I$(CUPLA_BASE)/include
export CUPLA_LDFLAGS := -L$(CUPLA_LIBDIR)
export CUPLA_NVCC_CXXFLAGS :=

KOKKOS_BASE := $(EXTERNAL_BASE)/kokkos
KOKKOS_SRC := $(KOKKOS_BASE)/source
KOKKOS_BUILD := $(KOKKOS_BASE)/build
export KOKKOS_INSTALL := $(KOKKOS_BASE)/install
KOKKOS_LIBDIR := $(KOKKOS_INSTALL)/lib
export KOKKOS_LIB := $(KOKKOS_LIBDIR)/libkokkoscore.a
KOKKOS_MAKEFILE := $(KOKKOS_BUILD)/Makefile
# For SERIAL to be enabled always, allow host-parallel and device-parallel to be (un)set
export KOKKOS_HOST_PARALLEL :=
export KOKKOS_DEVICE_PARALLEL := CUDA
KOKKOS_CUDA_ARCH := 70
ifeq ($(KOKKOS_CUDA_ARCH),50)
  KOKKOS_CMAKE_CUDA_ARCH := -DKokkos_ARCH_MAXWELL50=On
else ifeq ($(KOKKOS_CUDA_ARCH),70)
  KOKKOS_CMAKE_CUDA_ARCH := -DKokkos_ARCH_VOLTA70=On
else ifeq ($(KOKKOS_CUDA_ARCH),75)
  KOKKOS_CMAKE_CUDA_ARCH := -DKokkos_ARCH_TURING75=On
else
  $(error Unsupported KOKKOS_CUDA_ARCH $(KOKKOS_CUDA_ARCH). Likely it is sufficient just add another case in the Makefile)
endif
KOKKOS_HIP_ARCH := VEGA900
ifeq ($(KOKKOS_HIP_ARCH),VEGA900)
  KOKKOS_CMAKE_HIP_ARCH := -DKokkos_ARCH_VEGA900=On
else ifeq ($(KOKKOS_HIP_ARCH),VEGA909)
  KOKKOS_CMAKE_HIP_ARCH := -DKokkos_ARCH_VEGA909=On
else
  $(error Unsupported KOKKOS_HIP_ARCH $(KOKKOS_HIP_ARCH). Likely it is sufficient just add another case in the Makefile)
endif
export KOKKOS_CXXFLAGS := -I$(KOKKOS_INSTALL)/include
$(eval $(call CUFLAGS_template,$(KOKKOS_CUDA_ARCH),KOKKOS_))
export KOKKOS_LDFLAGS := -L$(KOKKOS_INSTALL)/lib -lkokkoscore -ldl
export KOKKOS_NVCC_CXXFLAGS :=
export NVCC_WRAPPER_DEFAULT_COMPILER := $(CXX)

KOKKOS_CMAKEFLAGS := -DCMAKE_INSTALL_PREFIX=$(KOKKOS_INSTALL) \
                     -DCMAKE_INSTALL_LIBDIR=lib \
                     -DKokkos_CXX_STANDARD=14 \
                     -DKokkos_ENABLE_SERIAL=On
KOKKOS_IS_SHARED :=
ifndef KOKKOS_DEVICE_PARALLEL
  KOKKOS_CMAKEFLAGS += -DCMAKE_CXX_COMPILER=g++
  export KOKKOS_DEVICE_CXX := $(CXX)
  export KOKKOS_DEVICE_CXX_NAME := GCC
  export KOKKOS_DEVICE_LDFLAGS := $(LDFLAGS)
  export KOKKOS_DEVICE_SO_LDFLAGS := $(SO_LDFLAGS)
  export KOKKOS_DEVICE_CXXFLAGS := $(CXXFLAGS)
  export KOKKOS_DEVICE_TEST_CXXFLAGS :=
else
  ifeq ($(KOKKOS_DEVICE_PARALLEL),CUDA)
    KOKKOS_CMAKEFLAGS += -DCMAKE_CXX_COMPILER=$(KOKKOS_SRC)/bin/nvcc_wrapper -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_CONSTEXPR=On -DKokkos_ENABLE_CUDA_LAMBDA=On -DKokkos_CUDA_DIR=$(CUDA_BASE) $(KOKKOS_CMAKE_CUDA_ARCH)
    export KOKKOS_DEVICE_CXX := $(CUDA_NVCC)
    export KOKKOS_DEVICE_CXX_NAME := NVCC
    export KOKKOS_DEVICE_LDFLAGS := $(LDFLAGS_NVCC)
    export KOKKOS_DEVICE_SO_LDFLAGS := $(SO_LDFLAGS_NVCC)
    export KOKKOS_DEVICE_CXXFLAGS := $(KOKKOS_NVCC_COMMON) $(CUDA_CXXFLAGS) $(USER_CUDAFLAGS)
    export KOKKOS_DEVICE_TEST_CXXFLAGS := $(CUDA_TEST_CXXFLAGS)
  else ifeq ($(KOKKOS_DEVICE_PARALLEL),HIP)
    KOKKOS_CMAKEFLAGS += -DCMAKE_CXX_COMPILER=$(ROCM_HIPCC) -DCMAKE_CXX_FLAGS="--gcc-toolchain=$(GCC_TOOLCHAIN)" -DKokkos_ENABLE_HIP=On $(KOKKOS_CMAKE_HIP_ARCH) -DBUILD_SHARED_LIBS=On
    KOKKOS_IS_SHARED := 1
    export KOKKOS_LIB := $(KOKKOS_LIBDIR)/libkokkoscore.so
    export KOKKOS_DEVICE_CXX := $(ROCM_HIPCC)
    export KOKKOS_DEVICE_CXX_NAME := HIPCC
    export KOKKOS_DEVICE_LDFLAGS := $(HIPCC_LDFLAGS)
    export KOKKOS_DEVICE_SO_LDFLAGS := $(SO_LDFLAGS)
    export KOKKOS_DEVICE_CXXFLAGS := $(HIPCC_CXXFLAGS)
    export KOKKOS_DEVICE_TEST_CXXFLAGS :=
  else
    $(error Unsupported KOKKOS_DEVICE_PARALLEL $(KOKKOS_DEVICE_PARALLEL))
  endif
endif
ifdef KOKKOS_HOST_PARALLEL
  ifeq ($(KOKKOS_HOST_PARALLEL),PTHREAD)
    KOKKOS_CMAKEFLAGS += -DKokkos_ENABLE_PTHREAD=On
    ifndef KOKKOS_PTHREAD_DISABLE_HWLOC
      KOKKOS_CMAKEFLAGS += -DKokkos_ENABLE_HWLOC=On -DKokkos_HWLOC_DIR=$(HWLOC_BASE)
      KOKKOS_CXXFLAGS += $(HWLOC_CXXFLAGS)
      KOKKOS_LDFLAGS += $(HWLOC_LDFLAGS)
    endif
  else
    $(error Unsupported KOKKOS_HOST_PARALLEL $(KOKKOS_HOST_PARALLEL))
  endif
endif
export KOKKOS_DEPS := $(KOKKOS_LIB)

# Intel oneAPI
ONEAPI_BASE := /opt/intel/oneapi
ONEAPI_ENV  := $(ONEAPI_BASE)/setvars.sh
DPCT_BASE   := $(ONEAPI_BASE)/dpcpp-ct/latest
SYCL_BASE   := $(ONEAPI_BASE)/compiler/latest/linux
SYCL_UNSUPPORTED_CXXFLAGS := --param vect-max-version-for-alias-checks=50 -Wno-non-template-friend -Werror=format-contains-nul -Werror=return-local-addr -Werror=unused-but-set-variable

# to use a different toolchain
#   - unset ONEAPI_ENV
#   - set SYCL_BASE appropriately

# check if libraries are under lib or lib64
ifdef SYCL_BASE
ifneq ($(wildcard $(SYCL_BASE)/lib/libsycl.so),)
SYCL_LIBDIR := $(SYCL_BASE)/lib
else ifneq ($(wildcard $(SYCL_BASE)/lib64/libsycl.so),)
SYCL_LIBDIR := $(SYCL_BASE)/lib64
else
SYCL_BASE :=
endif
endif
ifdef SYCL_BASE
export SYCL_CXX      := $(SYCL_BASE)/bin/dpcpp
export SYCL_CXXFLAGS := -fsycl -I$(DPCT_BASE)/include $(filter-out $(SYCL_UNSUPPORTED_CXXFLAGS),$(CXXFLAGS))
ifdef CUDA_BASE
export SYCL_CUDA_PLUGIN := $(wildcard $(SYCL_LIBDIR)/libpi_cuda.so)
export SYCL_CUDA_FLAGS  := --cuda-path=$(CUDA_BASE) -Wno-unknown-cuda-version
endif
endif

# force the recreation of the environment file any time the Makefile is updated, before building any other target
-include environment

# Targets and their dependencies on externals
TARGETS_ALL := $(notdir $(wildcard $(SRC_DIR)/*))
define TARGET_ALL_DEPS_template
include src/$(1)/Makefile.deps
endef
$(foreach target,$(TARGETS_ALL),$(eval $(call TARGET_ALL_DEPS_template,$(target))))

# Split targets by required toolchain
TARGETS_CUDA :=
TARGETS_HIP :=
TARGETS_SYCL :=
define SPLIT_TARGETS_template
ifneq ($$(filter $(1),$$($(2)_EXTERNAL_DEPENDS)),)
  TARGETS_$(1) += $(2)
endif
endef
TOOLCHAINS := CUDA HIP SYCL
$(foreach toolchain,$(TOOLCHAINS),$(foreach target,$(TARGETS_ALL),$(eval $(call SPLIT_TARGETS_template,$(toolchain),$(target)))))

TARGETS_GCC := $(filter-out $(TARGETS_CUDA) $(TARGETS_HIP) $(TARGETS_SYCL),$(TARGETS_ALL))

# Re-construct targets based on available compilers/toolchains
TARGETS := $(TARGETS_GCC)
ifdef CUDA_BASE
TARGETS += $(TARGETS_CUDA)
endif
ifdef ROCM_BASE
TARGETS += $(TARGETS_HIP)
endif
ifdef SYCL_BASE
TARGETS += $(TARGETS_SYCL)
endif
# remove possible duplicates
TARGETS := $(sort $(TARGETS))
all: $(TARGETS)

# Define test targets for each architecture
TEST_TARGETS := $(patsubsts %,test_%,$(TARGETS))
TEST_CPU_TARGETS := $(patsubst %,test_%_cpu,$(TARGETS))
TEST_NVIDIAGPU_TARGETS := $(patsubst %,test_%_nvidiagpu,$(TARGETS))
TEST_AMDGPU_TARGETS := $(patsubst %,test_%_amdgpu,$(TARGETS))
TEST_INTELGPU_TARGETS := $(patsubst %,test_%_intelgpu,$(TARGETS))
TEST_AUTO_TARGETS := $(patsubst %,test_%_auto,$(TARGETS))
test: test_cpu test_auto
ifdef CUDA_BASE
test: test_nvidiagpu
endif
ifdef ROCM_BASE
test: test_amdgpu
endif
ifdef SYCL_BASE
test: test_intelgpu
endif
test_cpu: $(TEST_CPU_TARGETS)
test_nvidiagpu: $(TEST_NVIDIAGPU_TARGETS)
test_amdgpu: $(TEST_AMDGPU_TARGETS)
test_intelgpu: $(TEST_INTELGPU_TARGETS)
test_auto: $(TEST_AUTO_TARGETS)
# $(TARGETS) needs to be PHONY because only the called Makefile knows their dependencies
.PHONY: all $(TARGETS)
.PHONY: test $(TEST_TARGETS)
.PHONY: test_cpu $(TEST_CPU_TARGETS)
.PHONY: test_nvidiagpu $(TEST_NVIDIAGPU_TARGETS)
.PHONY: test_amdgpu $(TEST_AMDGPU_TARGETS)
.PHONY: test_intelgpu $(TEST_INTELGPU_TARGETS)
.PHONY: test_auto $(TEST_AUTO_TARGETS)
.PHONY: format $(patsubst %,format_%,$(TARGETS_ALL))
.PHONY: environment print_targets clean distclean dataclean
.PHONY: external_tbb external_cub external_eigen external_kokkos external_kokkos_clean

environment: env.sh
env.sh: Makefile
	@echo '#! /bin/bash'                                                    >  $@
	@echo 'if [ -f .original_env ]; then'                                   >> $@
	@echo '  source .original_env'                                          >> $@
	@echo 'else'                                                            >> $@
	@echo '  echo "#! /bin/bash"                       >  .original_env'    >> $@
	@echo '  echo "PATH=$$PATH"                         >> .original_env'   >> $@
	@echo '  echo "LD_LIBRARY_PATH=$$LD_LIBRARY_PATH"   >> .original_env'   >> $@
	@echo 'fi'                                                              >> $@
	@echo                                                                   >> $@
	@echo -n 'export LD_LIBRARY_PATH='                                      >> $@
	@echo -n '$(TBB_LIBDIR):'                                               >> $@
	@echo -n '$(BACKTRACE_BASE)/lib:'                                       >> $@
ifeq ($(NEED_BOOST),true)
	@echo -n '$(BOOST_BASE)/lib:'                                           >> $@
endif
	@echo -n '$(HWLOC_BASE)/lib:'                                           >> $@
ifdef CUDA_BASE
	@echo -n '$(CUDA_LIBDIR):'                                              >> $@
endif
	@echo -n '$(CUPLA_LIBDIR):'                                             >> $@
	@echo -n '$(KOKKOS_LIBDIR):'                                            >> $@
ifneq ($(SYCL_BASE),)
ifeq ($(wildcard $(ONEAPI_ENV)),)
	@echo -n '$(SYCL_LIBDIR):'                                              >> $@
endif
endif
	@echo '$$LD_LIBRARY_PATH'                                               >> $@
	@echo -n 'export PATH='                                                 >> $@
ifdef CUDA_BASE
	@echo -n '$(CUDA_BASE)/bin:'                                            >> $@
endif
ifneq ($(SYCL_BASE),)
ifeq ($(wildcard $(ONEAPI_ENV)),)
	@echo -n '$(SYCL_BASE)/bin:'                                            >> $@
endif
endif
	@echo '$$PATH'                                                          >> $@
# check if oneAPI environment file exists
ifneq ($(wildcard $(ONEAPI_ENV)),)
	@echo 'source $(ONEAPI_ENV)'                                            >> $@
endif

define TARGET_template
$(1): $$(foreach dep,$$($(1)_EXTERNAL_DEPENDS),$$($$(dep)_DEPS)) | $(DATA_DEPS)
	+$(MAKE) -C src/$(1)

test_$(1): test_$(1)_cpu test_$(1)_auto
ifdef CUDA_BASE
test_$(1): test_$(1)_nvidiagpu
endif
ifdef ROCM_BASE
test_$(1): test_$(1)_amdgpu
endif
ifdef SYCL_BASE
test_$(1): test_$(1)_intelgpu
endif

test_$(1)_cpu: $(1)
	@echo
	@echo "Testing $(1) for CPU"
	+$(MAKE) -C src/$(1) test_cpu
	@echo

test_$(1)_nvidiagpu: $(1)
	@echo
	@echo "Testing $(1) for NVIDIA GPU device"
	+$(MAKE) -C src/$(1) test_nvidiagpu
	@echo

test_$(1)_amdgpu: $(1)
	@echo
	@echo "Testing $(1) for AMD GPU device"
	+$(MAKE) -C src/$(1) test_amdgpu
	@echo

test_$(1)_intelgpu: $(1)
	@echo
	@echo "Testing $(1) for Intel GPU device"
	+$(MAKE) -C src/$(1) test_intelgpu
	@echo

test_$(1)_auto: $(1)
	@echo
	@echo "Testing $(1) for automatic device selection"
	+$(MAKE) -C src/$(1) test_auto
	@echo
endef
$(foreach target,$(TARGETS),$(eval $(call TARGET_template,$(target))))

print_targets:
	@echo "Following program targets are available"
	@echo $(TARGETS)

define FORMAT_template
format_$(1):
	@echo "Formatting $(1)"
	@$(CLANG_FORMAT) -i $$(shell find $(SRC_DIR)/$(1) -name "*.h" -o -name "*.cc" -o -name "*.cu")
endef
$(foreach target,$(TARGETS_ALL),$(eval $(call FORMAT_template,$(target))))

format: $(patsubst %,format_%,$(TARGETS_ALL))

clean:
	rm -fR lib obj test $(TARGETS_ALL)

distclean: | clean
	rm -fR external .original_env

dataclean:
	rm -fR data/*.tar.gz data/*.bin data/data_ok

define CLEAN_template
clean_$(1):
	rm -fR lib/$(1) obj/$(1) test/$(1) $(1)
endef
$(foreach target,$(TARGETS_ALL),$(eval $(call CLEAN_template,$(target))))

# Data rules
$(DATA_DEPS): $(DATA_TAR_GZ) | $(DATA_BASE)/md5.txt
	cd $(DATA_BASE) && tar zxf $(DATA_TAR_GZ)
	cd $(DATA_BASE) && md5sum *.bin | diff -u md5.txt -
	touch $(DATA_DEPS)

$(DATA_TAR_GZ): | $(DATA_BASE)/url.txt
	curl -L -s -S $(shell cat $(DATA_BASE)/url.txt) -o $@

# External rules
$(EXTERNAL_BASE):
	mkdir -p $@

# TBB
external_tbb: $(TBB_LIB)

$(TBB_BASE):
	git clone --branch 2019_U9 https://github.com/intel/tbb.git $@

$(TBB_LIBDIR): $(TBB_BASE)
	mkdir -p $@

# Let TBB Makefile to define its own CXXFLAGS
$(TBB_LIB): CXXFLAGS:=
$(TBB_LIB): $(TBB_BASE) $(TBB_LIBDIR)
	+$(MAKE) -C $(TBB_BASE) stdver=c++17
	cp $$(find $(TBB_BASE)/build -name *.so*) $(TBB_LIBDIR)

# Eigen
external_eigen: $(EIGEN_BASE)

$(EIGEN_BASE):
	# from Eigen master branch as of 2021.08.18
	git clone -b cms/master/82dd3710dac619448f50331c1d6a35da673f764a https://github.com/cms-externals/eigen-git-mirror.git $@
	# include all Patatrack updates
	cd $@ && git reset --hard 733e6166b2f8b4edd23da33985187fd60903e9ca

# Boost
.PHONY: external_boost
external_boost: $(BOOST_BASE)

# Let Boost define its own CXXFLAGS
$(BOOST_BASE): CXXFLAGS:=
$(BOOST_BASE):
	$(eval BOOST_TMP := $(shell mktemp -d))
	curl -L -s -S https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.bz2 | tar xj -C $(BOOST_TMP)
	cd $(BOOST_TMP)/boost_1_76_0 && ./bootstrap.sh && ./b2 install --prefix=$@ --without-graph_parallel --without-mpi --without-python
	@rm -rf $(BOOST_TMP)
	$(eval undefine BOOST_TMP)

# libbacktrace
.PHONY: external_libbacktrace
external_libbacktrace: $(BACKTRACE_BASE)

# Let libbacktrace define its own CXXFLAGS
$(BACKTRACE_BASE): CXXFLAGS:=
$(BACKTRACE_BASE):
	$(eval BACKTRACE_TMP := $(shell mktemp -d))
	git clone https://github.com/ianlancetaylor/libbacktrace.git $(BACKTRACE_TMP)
	cd $(BACKTRACE_TMP)/ && ./configure --prefix=$@ --enable-shared
	$(MAKE) -C $(BACKTRACE_TMP)
	$(MAKE) -C $(BACKTRACE_TMP) install
	@rm -rf $(BACKTRACE_TMP)
	$(eval undefine BACKTRACE_TMP)

# hwloc
.PHONY: external_hwloc
external_hwloc: $(HWLOC_BASE)

# Let hwloc define its own CXXFLAGS
$(HWLOC_BASE): CXXFLAGS:=
$(HWLOC_BASE):
	$(eval HWLOC_TMP := $(shell mktemp -d))
	git clone -b hwloc-2.5.0 https://github.com/open-mpi/hwloc.git $(HWLOC_TMP)
	cd $(HWLOC_TMP)/ && ./autogen.sh
	cd $(HWLOC_TMP)/ && ./configure --prefix=$@ --enable-shared
	$(MAKE) -C $(HWLOC_TMP)
	$(MAKE) -C $(HWLOC_TMP) install
	@rm -rf $(HWLOC_TMP)
	$(eval undefine HWLOC_TMP)

# Alpaka
.PHONY: external_alpaka
external_alpaka: $(ALPAKA_BASE)

$(ALPAKA_BASE):
	git clone git@github.com:alpaka-group/alpaka.git -b 0.7.0 $@

# Cupla
.PHONY: external_cupla
external_cupla: $(CUPLA_BASE)/lib

$(CUPLA_BASE):
	git clone git@github.com:alpaka-group/cupla.git -b 0.3.0 $@
	cd $@ && git config core.sparsecheckout true && /usr/bin/echo -e '/*\n!/alpaka\n!/build\n!/lib' > .git/info/sparse-checkout && git read-tree -v -mu HEAD

$(CUPLA_BASE)/lib: $(CUPLA_BASE) $(ALPAKA_DEPS) $(BOOST_DEPS) $(TBB_DEPS) $(CUDA_DEPS)
	$(MAKE) -C $(CUPLA_BASE) -f $(BASE_DIR)/Makefile.cupla CXX=$(CXX) CUDA_BASE=$(CUDA_BASE) BOOST_BASE=$(BOOST_BASE) TBB_BASE=$(TBB_BASE) ALPAKA_BASE=$(ALPAKA_BASE)

# Kokkos
external_kokkos: $(KOKKOS_LIB)

$(KOKKOS_SRC):
	git clone --branch 3.4.01 https://github.com/kokkos/kokkos.git $@

$(KOKKOS_BUILD):
	mkdir -p $@

ifeq ($(KOKKOS_HOST_PARALLEL),PTHREAD)
ifndef KOKKOS_PTHREAD_DISABLE_HWLOC
$(KOKKOS_MAKEFILE): $(HWLOC_BASE)
endif
endif
$(KOKKOS_MAKEFILE): $(KOKKOS_SRC) | $(KOKKOS_BUILD)
	cd $(KOKKOS_BUILD) && $(CMAKE) $(KOKKOS_SRC) $(KOKKOS_CMAKEFLAGS)

# Let Kokkos' generated Makefile to define its own CXXFLAGS
# Except we need -fPIC (!)
$(KOKKOS_LIB): CXXFLAGS:=-fPIC
$(KOKKOS_LIB): $(KOKKOS_MAKEFILE)
	$(MAKE) -C $(KOKKOS_BUILD)
	$(MAKE) -C $(KOKKOS_BUILD) install

external_kokkos_clean:
	rm -fR $(KOKKOS_BUILD) $(KOKKOS_INSTALL)
