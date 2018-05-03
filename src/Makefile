.SUFFIXES:  .cpp .cu .o
CUDA_HOME := /usr/local/cuda
INC := -I$(CUDA_HOME)/include -I ../inc/
LIB := -L$(CUDA_HOME)/lib -lcudart
CC  := nvcc
DEP :=
ifeq ($(def), )
DEF :=
else
DEF := -D$(def)
endif

CCFLAGS := -std=c++11
NVCCFLAGS   := -lineinfo -arch=sm_53 --ptxas-options=-v -g
DYNCUDAFLAGS := -arch=compute_35 -rdc=true -lcudadevrt

all: main

huffman: huffman.o
		$(CC) $(INC) $(DYNCUDAFLAGS) $(CCFLAGS) -o huffman.o $(LIB)

main: main.o huffman.o histogram.o binary_tree.o $(DEP)
		$(CC) $(INC) $(DYNCUDAFLAGS) $(CCFLAGS) -o main main.o huffman.o histogram.o binary_tree.o $(LIB)

.cpp.o:
		$(CC) $(INC) $(NVCCFLAGS) $(CCFLAGS) $(DEF) -c $< -o $@

.cu.o:
		$(CC) $(INC) $(DYNCUDAFLAGS) $(CCFLAGS) $(DEF) -c $< -o $@

clean:
		rm -f *.o main

