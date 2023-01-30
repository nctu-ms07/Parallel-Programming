#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 8
#define STREAM_NUM 15

__global__ void mandelKernel(const float x0, const float y0,
                             const float dx, const float dy,
                             const int width, const int row_offset,
                             const int count,
                             int *const d_img) {
  // To avoid error caused by the floating number, use the following pseudo code
  //
  // float x = lowerX + thisX * stepX;
  // float y = lowerY + thisY * stepY;

  int loop_i = blockIdx.x * blockDim.x + threadIdx.x;
  int loop_j = blockIdx.y * blockDim.y + threadIdx.y + row_offset;

  const float2 c = make_float2(x0 + loop_i * dx, y0 + loop_j * dy);
  float2 z = c;
  float2 new_z;

  int i;
  for (i = 0; i < count; ++i) {

    if (z.x * z.x + z.y * z.y > 4.f)
      break;

    new_z.x = z.x * z.x - z.y * z.y;
    new_z.y = 2.f * z.x * z.y;
    z.x = c.x + new_z.x;
    z.y = c.y + new_z.y;
  }

  // int index = (loop_j * width + loop_i);
  d_img[loop_j * width + loop_i] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  const int size = resX * resY * sizeof(int);
  cudaHostRegister(img, size, cudaHostRegisterDefault);
  int *d_img;
  cudaHostGetDevicePointer(&d_img, img, 0);

  cudaStream_t streams[STREAM_NUM];
  for (int i = 0; i < STREAM_NUM; i++) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
  }

  const int stream_row_size = resY / STREAM_NUM;

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(resX / block.x, stream_row_size / block.y);

  int row_offset = 0;
  for (int i = 0; i < STREAM_NUM; i++) {
    mandelKernel<<<grid, block, 0, streams[i]>>>(lowerX, lowerY, stepX, stepY, resX, row_offset, maxIterations, d_img);
    row_offset += stream_row_size;
  }

  cudaDeviceSynchronize();
  for (int i = 0; i < STREAM_NUM; i++) {
    cudaStreamDestroy(streams[i]);
  }
  cudaHostUnregister(img);
}
