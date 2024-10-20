#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024 // square matrix width
#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

/* function declarations */
__global__ void matMul(const float *A, const float *B, float *C, int n);
void cpuMatMul(const float *A, const float *B, float *C, int n);
void initRand(float *A, int n);
void cudaCheck(cudaError_t err, const char *file, int line);

/* driver program */
int main(void) {
  srand(time(NULL));                   // initialize rng
  float *A_h, *B_h, *C_h, *C_cpu;      // host arrays
  float *A_d, *B_d, *C_d;              // device arrays
  size_t size = N * N * sizeof(float); // array size in bytes

  // allocate host memory
  A_h = (float *)malloc(size);
  B_h = (float *)malloc(size);
  C_h = (float *)malloc(size);
  C_cpu = (float *)malloc(size);

  // allocate device memory
  CUDA_CHECK(cudaMalloc(&A_d, size));
  CUDA_CHECK(cudaMalloc(&B_d, size));
  CUDA_CHECK(cudaMalloc(&C_d, size));

  // init input host arrays
  printf("Initializing A and B on host...\n"); // TODO: diagnostic prints
  initRand(A_h, N);
  initRand(B_h, N);

  // copy input arrays from host to device
  printf("Copying A and B from host to device...\n"); // TODO: diagnostic prints
  CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

  // launch kernel for simple vector addition
  printf("Computing matmul on GPU...\n"); // TODO: diagnostic prints
  dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
  dim3 blocksPerGrid(ceil(N / float(threadsPerBlock.x)), 1, 1);
  matMul<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Matmul on GPU complete.\n"); // TODO: diagnostic prints

  // copy output array from device to host
  CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

  printf("Computing matmul on CPU...\n"); // TODO: diagnostic prints
  cpuMatMul(A_h, B_h, C_cpu, N);
  printf("Matmul on CPU complete.\n"); // TODO: diagnostic prints

  // verify result
  for (int i = 0; i < 5; ++i) {
    printf("C_gpu[%d] = %f, C_cpu[%d] = %f\n", i, C_h[i], i, C_cpu[i]);
  }

  // clean up memory
  printf("Cleaning up memory...\n"); // TODO: diagnostic prints
  free(A_h);
  free(B_h);
  free(C_h);
  free(C_cpu);
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

  fprintf(stdout, "VECTOR ADDITION PROGRAM COMPLETE.\n");
  return 0;
}

/* matrix multiplication */
__global__ void matMul(const float *A, const float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
      acc += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = acc;
  }
}

/* matrix multiplication on the cpu */
void cpuMatMul(const float *A, const float *B, float *C, int n) {
  for (int i = 0; i < n; ++i) {     // i-th row of A
    for (int j = 0; j < n; ++j) {   // j-th col of B
      for (int k = 0; k < n; ++k) { // k-th element of corresponding row and col
        C[i * n + j] += A[i * n + k] + B[k * n + j];
      }
    }
  }
}

/* randomly initalize float array */
void initRand(float *a, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = (float)rand() / (float)RAND_MAX;
  }
}

/* cuda error handling */
void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
