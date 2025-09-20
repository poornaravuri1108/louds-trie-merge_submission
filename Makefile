CXX       ?= g++
NVCC      ?= nvcc

CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra

GENCODE_SM_LIST := 70 75 80 86 90
GENCODE_FLAGS   := $(foreach sm,$(GENCODE_SM_LIST),-gencode arch=compute_$(sm),code=sm_$(sm))

NVCCFLAGS := -O3 -std=c++17 -DUSE_CUDA $(GENCODE_FLAGS) \
             -Xcompiler="-Wall -Wextra -mbmi2"

# If your CUDA headers/libs are not on default paths, uncomment and adjust:
# CUDA_PATH ?= /usr/local/cuda
# NVCC      := $(CUDA_PATH)/bin/nvcc
# NVCCFLAGS += -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64

.PHONY: all cpu gpu clean

all: cpu gpu

cpu: merge-cpu

gpu: merge-gpu

merge-cpu: test.cpp louds-trie.cpp
	$(CXX)  $(CXXFLAGS)  $^ -o $@

merge-gpu: test.cpp louds-trie.cpp
	$(NVCC) $(NVCCFLAGS) $^ -o $@

clean:
	rm -f merge-cpu merge-gpu *.o
