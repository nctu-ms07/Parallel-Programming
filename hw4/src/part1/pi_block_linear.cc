#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv) {
  // --- DON'T TOUCH ---
  MPI_Init(&argc, &argv);
  double start_time = MPI_Wtime();
  double pi_result;
  long long int tosses = atoi(argv[1]);
  int world_rank, world_size;
  // ---

  // TODO: init MPI
  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  long long toss_num = tosses / world_size;
  long long in_circle_num = 0;

  unsigned int seed = (world_rank + 1) * time(nullptr);

  for (long long i = 0; i < toss_num; i++) {
    float x = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
    float y = rand_r(&seed) / ((float) RAND_MAX) * 2 - 1;
    if (x * x + y * y <= 1) {
      in_circle_num++;
    }
  }

  if (world_rank > 0) {
    // TODO: handle workers
    MPI_Send(&in_circle_num, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD);
  } else if (world_rank == 0) {
    // TODO: master
    long long in_circle_num_received;
    for (int i = 1; i < world_size; i++) {
      MPI_Recv(&in_circle_num_received,
               1,
               MPI_LONG_LONG_INT,
               MPI_ANY_SOURCE,
               MPI_ANY_TAG,
               MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      in_circle_num += in_circle_num_received;
    }
  }

  if (world_rank == 0) {
    // TODO: process PI result
    pi_result = 4 * (in_circle_num / ((double) tosses));

    // --- DON'T TOUCH ---
    double end_time = MPI_Wtime();
    printf("%lf\n", pi_result);
    printf("MPI running time: %lf Seconds\n", end_time - start_time);
    // ---
  }

  MPI_Finalize();
  return 0;
}
