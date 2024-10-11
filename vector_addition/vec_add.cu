/* simple vector addition */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

/* number of elements in each array (object) */
#define N 1000000

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
void rand_init(float *A, unsigned n) {
  for (unsigned i = 0; i < n; ++i) {
    A[i] = (float)rand() / RAND_MAX;
  }
}

/* initialize array with a given float */
void val_init(float *A, unsigned n, float value) {
  for (unsigned i = 0; i < n; ++i) {
    A[i] = value;
  }
}

/* vector addition kernel */
__global__ void vecAddKernel(const float *A, const float *B, float *C,
                             unsigned n) {
  /* calculate global index for each thread */
  unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

/* vector addition kernel; each thread processes two adjacent array elements */
__global__ void vecAddAdjKernel(const float *A, const float *B, float *C,
                                unsigned n) {
  /* global address for each thread */
  unsigned i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;

  /* process first adjacent element */
  if (i < n) {
    C[i] = A[i] + B[i];
  }

  /* process second adjacent element */
  if (i + 1 < n) {
    C[i + 1] = A[i + 1] + B[i + 1];
  }
}

/* vector addition kernel; each thread processes two adjacent sections */
__global__ void vecAddSecKernel(const float *A, const float *B, float *C,
                                unsigned n) {
  /* global address for each thread:
   * each section is separated by `blockDim.x` elements
   */
  unsigned i = (blockDim.x * blockIdx.x * 2) + threadIdx.x;

  /* process element from first section */
  if (i < n) {
    C[i] = A[i] + B[i];
  }

  /* process element from second section */
  unsigned j = i + blockDim.x;
  if (j < n) {
    C[j] = A[j] + B[j];
  }
}

int main(void) {
  /* seed rng */
  srand(time(NULL));

  /* size of each object in bytes */
  unsigned size = N * sizeof(float);

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
  rand_init(A_h, N);
  rand_init(B_h, N);

  /* copy operand objects from host to device */
  CUDA_CHECK(cudaMemcpy(A_d, A_h, N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, N, cudaMemcpyHostToDevice));

  /* launch config */
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  /* cuda events for timing */
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  /* warmup run for vecAddKernel */
  vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  const int numRuns = 10;
  float totalTime = 0.0f;

  /* perform timed runs for vecAddKernel */
  for (int i = 0; i < numRuns; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* calculate and record elapsed time for kernel execution */
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    totalTime += ms;
  }

  float avgTime = totalTime / numRuns;

  fprintf(stdout,
          "vecAddKernel() took %f seconds on average to add two vectors of "
          "length %d.\n",
          avgTime, N);

  /* warmup run for vecAddAdjKernel */
  vecAddAdjKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  totalTime = 0.0f;

  /* perform timed runs for vecAddAdjKernel */
  for (int i = 0; i < numRuns; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    vecAddAdjKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* calculate and record elapsed time for kernel execution */
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    totalTime += ms;
  }

  avgTime = totalTime / numRuns;

  fprintf(stdout,
          "vecAddAdjKernel() took %f seconds on average to add two vectors of "
          "length %d.\n",
          avgTime, N);

  /* warmup run for vecAddSecKernel */
  vecAddSecKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  totalTime = 0.0f;

  /* perform timed runs for vecAddSecKernel */
  for (int i = 0; i < numRuns; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    vecAddSecKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* calculate and record elapsed time for kernel execution */
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    totalTime += ms;
  }

  avgTime = totalTime / numRuns;

  fprintf(stdout,
          "vecAddSecKernel() took %f seconds on average to add two "
          "vectors of length %d.\n",
          avgTime, N);

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

  fprintf(stdout, "PROGRAM COMPLETE.\n");
  return 0;
}
