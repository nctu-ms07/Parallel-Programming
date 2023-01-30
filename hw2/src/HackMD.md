## Q1
> In your write-up, produce a graph of speedup compared to the reference sequential implementation as a function of the number of threads used FOR VIEW 1.

```cpp=
  int totalRows = args->height / args->numThreads;
  int startRow = args->threadId * totalRows;
  if (args->threadId == args->numThreads - 1) {
    totalRows += args->height % args->numThreads;
  }

  mandelbrotSerial(args->x0,
                   args->y0,
                   args->x1,
                   args->y1,
                   args->width,
                   args->height,
                   startRow,
                   totalRows,
                   args->maxIterations,
                   args->output);
```


![](https://i.imgur.com/XHVDXhs.png)

> Is speedup linear in the number of threads used?

> Definitely not!

> In your writeup hypothesize why this is (or is not) the case?

```cpp=
static inline int mandel(float c_re, float c_im, int count) {
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) {

    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re * z_re - z_im * z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

```
> In function `mandel()`, we can see that the iteration count is bound by both `count` and `if (z_re * z_re + z_im * z_im > 4.f)`. Since `count` is always the same in this scenario, the only difference must lie in `if (z_re * z_re + z_im * z_im > 4.f)`. Luckly, the ouput is exactly the iteration count, which lies between 0 to 255, and it can be interpreted as the color in the mandelbrot graph. (0 for black and 255 for white). As a result, we can assume that if a thread is assigned for the brighter part, it consumes more time than others, which could possibly become the threshold of the execution time.

---

## Q2
> How do your measurements explain the speedup graph you previously created?

![](https://i.imgur.com/K1R6MaT.jpg)

```shell!
[Thread 0]:		[120.224] ms
[Thread 2]:		[124.483] ms
[Thread 1]:		[343.603] ms
[Thread 0]:		[110.302] ms
[Thread 2]:		[111.562] ms
[Thread 1]:		[361.198] ms
[Thread 2]:		[109.052] ms
[Thread 0]:		[112.283] ms
[Thread 1]:		[341.315] ms
[Thread 0]:		[110.385] ms
[Thread 2]:		[112.875] ms
[Thread 1]:		[345.139] ms
[Thread 2]:		[110.083] ms
[Thread 0]:		[114.224] ms
[Thread 1]:		[335.875] ms
```

> As I predicted, since thread 1 among 3 threads is outputting the birghter part of the image, it's execution time is much slower that the others. Also, due to the output image of thread 0 and 2 is symmetrical, their execution time is nearly the same.

---

## Q3
> In your write-up, describe your approach to parallelization and report the final 4-thread speedup obtained.

```cpp=
  for (unsigned int startRow = args->threadId; startRow < args->height; startRow += args->numThreads) {
    mandelbrotSerial(args->x0,
                     args->y0,
                     args->x1,
                     args->y1,
                     args->width,
                     args->height,
                     startRow,
                     1,
                     args->maxIterations,
                     args->output);
  }
```

> The method tries to spilts the workload evenly among threads by assigning rows one by one to different thread on round robin bases.

![](https://i.imgur.com/SS6wLa4.png)


---

## Q4
> Now run your improved code with eight threads. Is performance noticeably greater than when running with four threads? (Notice that the workstation server provides 4 cores 4 threads.)

> ![](https://i.imgur.com/Q8TR1cW.png)

> Not even close!

> Why or why not?

> Since there's only 4 threads in the workstation server, the maximum thread running in parallelism is 4. Therefore, if there are more than 4 threads, it can only be executed through context switch, which causes overhead and requires more time.


