NVCC = nvcc
CXX = CC
NVCC_FLAGS = -O3 -arch=sm_80 -Xcompiler -fPIC
CXX_FLAGS = -O3 -std=c++17
CUDA_PATH ?= $(or $(CUDA_HOME),$(CUDATOOLKIT_HOME))

ifeq ($(strip $(CUDA_PATH)),)
$(error Set CUDA_HOME or CUDATOOLKIT_HOME before running make)
endif

INCLUDES = -I$(CUDA_PATH)/include
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas

TARGET = pipeline_dist_attn
OBJS = flash_attn.o main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXX_FLAGS) -o $(TARGET) $(OBJS) $(LIBS)

flash_attn.o: flash_attn.cu flash_attn.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c flash_attn.cu -o flash_attn.o

main.o: main.cpp flash_attn.cuh
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c main.cpp -o main.o

clean:
	rm -f $(OBJS) $(TARGET)