# 默认取 nvcc 所在的 CUDA 安装目录，可用 make CUDA_DIR=... 覆盖
CUDA_DIR ?= $(patsubst %/bin/nvcc,%,$(shell which nvcc))

CC_SRCS := $(wildcard src/*.cc)
CC_OBJS := ${CC_SRCS:src/%.cc=build/obj/%.o}
CUDA_SRCS := $(wildcard src/*.cu)
CUDA_OBJS := ${CUDA_SRCS:src/%.cu=build/obj/%.o}
OBJS := $(CC_OBJS) $(CUDA_OBJS)

CC = g++
WARNINGS = -Wall -Wfatal-errors -Wno-unused -Wno-unused-result
CC_FLAGS = -std=c++11 -fPIC $(WARNINGS) -I$(CUDA_DIR)/include
LD_FLAGS = -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcublas

NVCC = nvcc
NVCC_FLAGS = -std=c++11 --compiler-options '-fPIC'
# CUDA 12 已不支持 compute_30/35；最后一项生成 PTX，供更新的 GPU（如 sm_120）JIT 运行
ARCH ?= -gencode arch=compute_60,code=sm_60 \
        -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 \
        -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86 \
        -gencode arch=compute_89,code=sm_89 \
        -gencode arch=compute_90,code=[sm_90,compute_90]

all: build/lib/libc_runtime_api.so

build/lib/libc_runtime_api.so: $(OBJS)
	@mkdir -p build/lib
	$(CC) -shared $^ -o $@ $(LD_FLAGS)

build/obj/%.o: src/%.cc
	@mkdir -p build/obj
	$(CC) $(CC_FLAGS) -c $< -o $@

build/obj/%.o: src/%.cu
	@mkdir -p build/obj
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf build

.PHONY: all clean