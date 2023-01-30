#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

__global__ void mandelKernel(const float x0, const float y0,
                             const float dx, const float dy,
                             const int width,
                             const int count,
                             int *const d_img,
                             const size_t pitch) {
  // To avoid error caused by the floating number, use the following pseudo code
  //
  // float x = lowerX + thisX * stepX;
  // float y = lowerY + thisY * stepY;

  int loop_i = blockIdx.x * blockDim.x + threadIdx.x;
  int loop_j = blockIdx.y * blockDim.y + threadIdx.y;

  float c_re = x0 + loop_i * dx;
  float c_im = y0 + loop_j * dy;

  float z_re = c_re, z_im = c_im;
  int i;
  float new_re, new_im;
  for (i = 0; i < count; ++i) {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    new_re = z_re * z_re - z_im * z_im;
    new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  // int index = (loop_j * width + loop_i);
  *((int *) ((char *) d_img + loop_j * pitch) + loop_i) = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int *img, int resX, int resY, int maxIterations) {
  float stepX = (upperX - lowerX) / resX;
  float stepY = (upperY - lowerY) / resY;

  const int size = resX * resY * sizeof(int);
  size_t pitch;
  int *h_img, *d_img;
  cudaHostAlloc(&h_img, size, cudaHostAllocDefault);
  cudaMallocPitch(&d_img, &pitch, resX * sizeof(int), resY);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(resX / block.x, resY / block.y);
  mandelKernel<<<grid, block>>>(lowerX, lowerY, stepX, stepY, resX, maxIterations, d_img, pitch);

  cudaMemcpy2D(h_img, resX * sizeof(int), d_img, pitch, resX * sizeof(int), resY, cudaMemcpyDefault);
  cudaFree(d_img);
  memcpy(img, h_img, size);
  cudaFreeHost(h_img);
}
