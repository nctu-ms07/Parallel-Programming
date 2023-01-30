## Q1
> Explain your implementation.

```cpp=
// create command queue
cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, NULL);
```

```cpp=
// allocate device memory and copy filter, inputImage to device memory
cl_mem d_filter = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, filterSize, filter, NULL);
cl_mem d_inputImage = clCreateBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, imageSize, inputImage, NULL);
cl_mem d_outputImage = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize, NULL, NULL);
```

```cpp=
// create kernel
cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);
// pass argument to kernel
clSetKernelArg(kernel, 0, sizeof(int), &filterWidth);
clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_inputImage);
clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_outputImage);
```

```cpp=
size_t global_work[] = {imageWidth, imageHeight};
size_t work_group[] = {GROUP_SIZE, GROUP_SIZE};
// execute kernel
clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work, work_group, 0, NULL, NULL);
```

```cpp=
// copy device memory to host
clEnqueueReadBuffer(command_queue, d_outputImage, CL_TRUE, 0, imageSize, outputImage, 0, NULL, NULL);
```

```cpp=
// release resource
clReleaseKernel(kernel);
clReleaseMemObject(d_filter);
clReleaseMemObject(d_inputImage);
clReleaseMemObject(d_outputImage);
clReleaseCommandQueue(command_queue);
```

> How do you optimize the performance of convolution?

> Use `__constant` address space for `filter`, and `restrict` qualifer to hint compiler to make optimizations (for example, vectorization) that would not otherwise have been possible.

```cpp=
__kernel void convolution(const int filterWidth,
                          __constant const float *restrict filter,
                          __global const float *restrict inputImage,
                          __global float *restrict outputImage);
```

> Calculate loop boundary outside of the loop

```cpp=
int row_offset_min = -min(y, halfFilterSize);
int row_offset_max = min(imageHeight - 1 - y, halfFilterSize);
int col_offset_min = -min(x, halfFilterSize);
int col_offset_max = min(imageWidth - 1 - x, halfFilterSize);
```

> Calculate `sum` using built-in math function `mad`

```cpp=
sum = mad(filter[filterBase + col_offset], inputImage[imageBase + col_offset], sum);
```
> Every pixel is handled by one `work item`. In the end, the kernel looks like this.

```cpp=
__kernel void convolution(const int filterWidth,
                          __constant const float *restrict filter,
                          __global const float *restrict inputImage,
                          __global float *restrict outputImage) {
  int imageWidth = get_global_size(0);
  int imageHeight = get_global_size(1);
  int halfFilterSize = filterWidth >> 1;

  int x = get_global_id(0);
  int y = get_global_id(1);

  int row_offset_min = -min(y, halfFilterSize);
  int row_offset_max = min(imageHeight - 1 - y, halfFilterSize);
  int col_offset_min = -min(x, halfFilterSize);
  int col_offset_max = min(imageWidth - 1 - x, halfFilterSize);

  float sum = 0;
  for (int row_offset = row_offset_min; row_offset <= row_offset_max; row_offset++) {
    int imageBase = (y + row_offset) * imageWidth + x;
    int filterBase = (halfFilterSize + row_offset) * filterWidth + halfFilterSize;
    for (int col_offset = col_offset_min; col_offset <= col_offset_max; col_offset++) {
      if (filter[filterBase + col_offset]) {
        sum = mad(filter[filterBase + col_offset], inputImage[imageBase + col_offset], sum);
      }
    }
  }

  outputImage[y * imageWidth + x] = sum;
}
```

---

## Q2
> Explain your CUDA implementation.

> The algorithm and optimize method is the same. However, since CUDA doesn't have `__constant` address space, I use `__shared__` address space instead. Also, the built-in math function changed to `__fmaf_rn`.

```cpp=
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
```

> Plot a chart to show the performance difference between using OpenCL and CUDA.

![](https://i.imgur.com/hYBuD4i.png)

> Explain the result.

> To my surprise, `OpenCL` performs better than `CUDA`. However, their implementation detail isn't exactly the same. For example, `OpenCL` simplely passes the `filter` into `__constant` address space, but we have to dynamic allocate `__shared__` memory in `CUDA` and copy `filter` from global memory to local memory. Also, the copying method in `CUDA` isn't specified clearly in the case of writing to device memory of pinned host memory. Lastly, the built-in math function is built differently. Therefore, it's hard to say which one is better.