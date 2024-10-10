/* simple vector addition */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define N 1000000 /* number of elements in each array (object) */

/* cuda error handling */
void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA ERROR] at file %s:%d\n%s\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

/* randomly initialize array */
void rand_init(float *A, long int n) {
  for (long int i = 0; i < n; ++i) {
    A[i] = (float)rand() / RAND_MAX;
  }
}

/* initialize array with a given float */
void val_init(float *A, long int n, float value) {
  for (long int i = 0; i < n; ++i) {
    A[i] = value;
  }
}

/* vector addition kernel */
__global__ void vecAddKernel(const float *A, const float *B, float *C,
                             long int n) {
  long int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

int main(void) {
  /* monotonically time the operation */
  struct timespec start_time, end_time;

  /* size of each object in bytes */
  long int size = N * sizeof(float);

  /* declare objects on host (CPU) */
  float *A_h, *B_h, *C_h;

  /* declare objects on device (GPU) */
  float *A_d, *B_d, *C_d;

  /* allocate memory for objects on host */
  A_h = (float *)malloc(size);
  B_h = (float *)malloc(size);
  C_h = (float *)malloc(size);

  /* allocate memory for objects on device */
  CUDA_CHECK(cudaMalloc((void **)&A_d, size));
  CUDA_CHECK(cudaMalloc((void **)&B_d, size));
  CUDA_CHECK(cudaMalloc((void **)&C_d, size));

  /* initialize host objects */
  /*
  rand_init(A_h, N);
  rand_init(B_h, N);
  */
  val_init(A_h, N, 1.0);
  val_init(B_h, N, 2.0);

  /* copy operand objects from host to device */
  CUDA_CHECK(cudaMemcpy(A_d, A_h, N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, N, cudaMemcpyHostToDevice));

  /* launch kernel with: N/256 blocks per grid and 256 threads per block */
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  vecAddKernel<<<ceil(N / 256.0), 256>>>(A_d, B_d, C_d, N);
  clock_gettime(CLOCK_MONOTONIC, &end_time);

  double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                        (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

  /* copy result object from device to host */
  CUDA_CHECK(cudaMemcpy(C_h, C_d, N, cudaMemcpyDeviceToHost));

  /* free host memory */
  free(A_h);
  free(B_h);
  free(C_h);

  /* free device memory */
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

  fprintf(stdout,
          "It took %f seconds to add two vectors of length %d in cuda.\n",
          elapsed_time, N);

  fprintf(stdout, "PROGRAM COMPLETE.\n");
  return 0;
}
