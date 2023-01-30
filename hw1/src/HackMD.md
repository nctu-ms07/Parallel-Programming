## Q1-1
#### VECTOR_WIDTH 2
![](https://i.imgur.com/RNesdBu.png)
#### VECTOR_WIDTH 4
![](https://i.imgur.com/dRJvLV9.png)
#### VECTOR_WIDTH 8
![](https://i.imgur.com/V0aAxuX.png)
#### VECTOR_WIDTH 16
![](https://i.imgur.com/JwOze9w.png)

> Does the vector utilization increase, decrease or stay the same as `VECTOR_WIDTH` changes? Why?

> Vector utilization decreases accroding to its definiation, as the possiblity of one vector lane to stay active in the process of multiplication in exponent operation is getting lower.

---

## Q2-1
> Fix the code to make sure it uses aligned moves for the best performance.

```cpp!
void test(float* __restrict a, float* __restrict b, float* __restrict c, int N) {
  __builtin_assume(N == 1024);
  // ----------------------------------------//
  a = (float *)__builtin_assume_aligned(a, 32);
  b = (float *)__builtin_assume_aligned(b, 32);
  c = (float *)__builtin_assume_aligned(c, 32);
  // ----------------------------------------//
  
  for (int i=0; i<I; i++) {
    for (int j=0; j<N; j++) {
      c[j] = a[j] + b[j];
    }
  }
}
```

---

## Q2-2
> What speedup does the vectorized code achieve over the unvectorized code?

> Almost 3 times faster according to the testing result below.

```shell!
# case 1
$ make clean && make && ./test_auto_vectorize -t 1
$ 6.36079sec (N: 1024, I: 20000000)
```

```shell!
# case 2
$ make clean && make VECTORIZE=1 && ./test_auto_vectorize -t 1
$ 2.24603sec (N: 1024, I: 20000000)
```

>What additional speedup does using `-mavx2` give (`AVX2=1` in the `Makefile`)?

> Almost 2 times faster according to the testing result below.

```shell!
# case 3
$ make clean && make VECTORIZE=1 AVX2=1 && ./test_auto_vectorize -t 1
$ 1.20165sec (N: 1024, I: 20000000)
```

> What can you infer about the bit width of the default vector registers on the PP machines?

> From the message `vectorized loop (vectorization width: 4, interleaved count: 2) [-Rpass=loop-vectorize]`, we can infer that there's 4 single-precision floating-point values with size 32 bits in vector register. Therefore, the bit width of the default vector register is 128 bits. (4 x 32 = 128)

>What about the bit width of the AVX2 vector registers.

> From the message `vectorized loop (vectorization width: 8, interleaved count: 4) [-Rpass=loop-vectorize]`, we can infer that there's 8 single-precision floating-point values with size 32 bits in vector register. Therefore, the bit width of the default vector register is 256 bits. (8 x 32 = 256)

---

## Q2-3
> Provide a theory for why the compiler is generating dramatically different assembly.

> According to the [compiled result](https://godbolt.org/z/zrv9Tr33q) of `test2.cpp.patch`, we can see that the compiler exploits `movaps` and `maxps` instructions to load, compare and get maximum value, and store in batch as every element in the array goes through the same operation, which isn't the case in `test2.cpp`. `c[j] = b[j]` are only for those who meets the requirement `b[j] > a[j]`, and it makes no sense to leave other elements staying in registers just to store in batch. 