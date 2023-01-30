## Q1
> 1\. How do you control the number of MPI processes on each node?

> Put `slot=#` after hostnames in the hostfile

> 2\. Which functions do you use for retrieving the rank of an MPI process and the total number of processes?

```cpp=
  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
```

---

## Q2
> 1\. Why `MPI_Send` and `MPI_Recv` are called “blocking” communication?

> Because both of them return only when the data transfer is  finished.

> 2\. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

![](https://i.imgur.com/wKjX35f.png)

---

## Q3
> 1\. Measure the performance (execution time) of the code for 2, 4, 8, 16 MPI processes and plot it.

![](https://i.imgur.com/gCaforB.png)

> 2\. How does the performance of binary tree reduction compare to the performance of linear reduction?

![](https://i.imgur.com/9QkgAhY.png)

> Almost the same. The difference is too small to say that one is faster than the other since there are other factors. (e.g. Running node is busy.)

> 3\. Increasing the number of processes, which approach (linear/tree) is going to perform better? Why? Think about the number of messages and their costs.

![](https://i.imgur.com/VfpkQAU.png)

> Almost the same. The difference is too small to say that one is faster than the other since there are other factors. (e.g. Running node is busy.) Also, it's hard to say which is better since linear reduction has lower data transfer frequency  but its reduction work load is heavier while binary tree reduction has higher data transfer frequency but its reduction work load is lighter.

---

## Q4
> 1\. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

![](https://i.imgur.com/BGXZvHe.png)

> 2\. What are the MPI functions for non-blocking communication?

> - **MPI_Isend**
> - **MPI_Irecv**
> - **MPI_Wait**
> - **MPI_Waitany**

> 3\. How the performance of non-blocking communication compares to the performance of blocking communication?

![](https://i.imgur.com/UzXSbjP.png)

> Almost the same. The difference is too small to say that one is faster than the other since there are other factors. (e.g. Running node is busy.)

---

## Q5
> 1\. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

![](https://i.imgur.com/wvTI6uu.png)

---

## Q6
> 1\. Measure the performance (execution time) of the code for 2, 4, 8, 12, 16 MPI processes and plot it.

![](https://i.imgur.com/h1H7m4Y.png)

---

## Q7
> 1\. Describe what approach(es) were used in your MPI matrix multiplication for each data set.

> I try to lower sending cost by only sending partial a_mat to workers so that every worker only needs to deal with `n / (world_size - 1)` rows. Note that master only takes care of the remainder `n % (worldsize - 1)` since master needs to send a_mat, b_mat and receive the result, I want to lower its work load. To lower cache misses while calculating, I send the transpose of b_mat so that access of column is cache friendly.





