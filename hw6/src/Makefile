default: conv

CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g

NVCC = nvcc
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3
CUDA_LINK_FLAGS =  -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -lOpenCL

OBJS = main.o bmpfuncs.o hostFE.o serialConv.o helper.o

conv: $(OBJS)
ifdef CUDA
	$(NVCC) -o $@ $(OBJS) $(CUDA_LINK_FLAGS)
else
	$(CC) -o $@ $(OBJS) $(FLAGS)
endif

%.o: %.cu
ifdef CUDA
	$(NVCC) -c $(CUDA_COMPILE_FLAGS) $< -o $@
endif

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

clean:
	rm -f conv *.o output.bmp ref.bmp