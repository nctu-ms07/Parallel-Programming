#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" {
#include "hostFE.h"
}

#define BLOCK_SIZE 8

__global__ void convolution(const int filterWidth,
                            const float *__restrict__ filter,
                            const float *__restrict__ inputImage,
                            float *__restrict__ outputImage) {
  extern __shared__ float shared_filter[];
  for (int i = 0; i < filterWidth * filterWidth; i++) {
    shared_filter[i] = filter[i];
  }
  __syncthreads();

  int imageWidth = blockDim.x * gridDim.x;
  int imageHeight = blockDim.y * gridDim.y;
  int halfFilterSize = filterWidth >> 1;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int row_offset_min = -min(y, halfFilterSize);
  int row_offset_max = min(imageHeight - 1 - y, halfFilterSize);
  int col_offset_min = -min(x, halfFilterSize);
  int col_offset_max = min(imageWidth - 1 - x, halfFilterSize);

  float sum = 0;
  for (int row_offset = row_offset_min; row_offset <= row_offset_max; row_offset++) {
    int imageBase = (y + row_offset) * imageWidth + x;
    int filterBase = (halfFilterSize + row_offset) * filterWidth + halfFilterSize;
    for (int col_offset = col_offset_min; col_offset <= col_offset_max; col_offset++) {
      if (shared_filter[filterBase + col_offset]) {
        sum = __fmaf_rn(shared_filter[filterBase + col_offset], inputImage[imageBase + col_offset], sum);
      }
    }
  }

  outputImage[y * imageWidth + x] = sum;
}

extern "C" void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
                       float *inputImage, float *outputImage, cl_device_id *device,
                       cl_context *context, cl_program *program) {
  int filterSize = filterWidth * filterWidth * sizeof(float);
  int imageSize = imageHeight * imageWidth * sizeof(float);

  float *d_filter;
  float *d_inputImage;
  float *d_outputImage;
  cudaMalloc(&d_filter, filterSize);
  cudaMalloc(&d_inputImage, imageSize);
  cudaHostRegister(outputImage, imageSize, cudaHostRegisterDefault);
  cudaHostGetDevicePointer(&d_outputImage, outputImage, 0);

  cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyDefault);
  cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyDefault);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(imageWidth / block.x, imageHeight / block.y);

  convolution<<<grid, block, filterSize * sizeof(float)>>>(filterWidth, d_filter, d_inputImage, d_outputImage);

  cudaDeviceSynchronize();

  cudaFree(d_filter);
  cudaFree(d_inputImage);
  cudaHostUnregister(outputImage);
}