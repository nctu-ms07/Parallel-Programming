## Q1
> What are the pros and cons of the three methods? Give an assumption about their performances.

> Assumption: Method 2 > Method 1 > Method 3

- Method 1
    - Pros
      -  Every pixel is handled by a thread.
      -  No paged-locked memory is required.
    - Cons
      - Some pixels required more time, and other pixels within the same warp need to wait.
      - Pagable memory may not always be ready for transferation.

- Method 2
    - Pros
        - Every pixel is handled by a thread.
        - Paged-locked memory is always ready for transferation.
        - Memory allocated by `cudaMallocPitch()` is always aligned, which is more efficient when it comes to transferation.
    - Cons
        - Some pixels required more time, and other pixels within the same warp need to wait.
        - Paged-locked memory is a limitted resource, and it may cause problem if it's insufficient for the operation system.
        - `cudaMallocPitch()` sometimes required extra memory for padding.

- Method 3
    - Pros
        - Paged-locked memory is always ready for transferation.
        - Memory allocated by `cudaMallocPitch()` is always aligned, which is more efficient when it comes to transferation.
        - It's more efficient to move data from local memory to global memory in GPU using `int4`.
    - Cons
        - Serial computation in each thread.
        - Paged-locked memory is a limitted resource, and it may cause problem if it's insufficient for the operation system.
        - `cudaMallocPitch()` sometimes required extra memory for padding.

---

## Q2
> How are the performances of the three methods? Plot a chart to show the differences among the three methods

> Performance for `View 1`

![](https://i.imgur.com/wZeZ1JY.png)

> Performance for `View 2`

![](https://i.imgur.com/si5ucPr.png)

---

## Q3
> Explain the performance differences thoroughly based on your experimental results. Does the results match your assumption? Why or why not.

> Since there's more white pixels in `View 1`, it's reasonable that the average time of `View 1` is always larger. And not surprisingly, `Method 3` is always the slowest since there's serial computation in each thread.

> No. Much to my surprise, `Method 1` is faster than `Method 2`. I guess it's because the size of the image isn't big enough so that the extra time for allocating padding memory overshadows the efficiency of transfering aligned memory from GPU to CPU.

---

## Q4
> Can we do even better? Think a better approach and explain it. Implement your method in `kernel4.cu`.

> Faster assignment using `float2`

```cpp=
  const float2 c = make_float2(x0 + loop_i * dx, y0 + loop_j * dy);
  float2 z = c;
  float2 new_z;
```

> Pin host memory that's already allocated, and get its device pointer, which is used in kernel function.

```cpp=
  const int size = resX * resY * sizeof(int);
  cudaHostRegister(img, size, cudaHostRegisterDefault);
  int *d_img;
  cudaHostGetDevicePointer(&d_img, img, 0);
```

> Hide transfer latency using multiple streams.

```cpp=
  const int stream_row_size = resY / STREAM_NUM;

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(resX / block.x, stream_row_size / block.y);

  int row_offset = 0;
  for (int i = 0; i < STREAM_NUM; i++) {
    mandelKernel<<<grid, block, 0, streams[i]>>>(lowerX, lowerY, stepX, stepY, resX, row_offset, maxIterations, d_img);
    row_offset += stream_row_size;
  }
```