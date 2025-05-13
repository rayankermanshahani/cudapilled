#include "../include/cuda_utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024 // square matrix width
#define THREADS_PER_BLOCK 256

/* function declarations */
__global__ void matVecProd(const float* A, const float* b, float* c, int n);
void cpuMatVecProd(const float* A, const float* b, float* c, int n);

/* driver program */
int main(void) {
  srand(time(NULL));                     // initialize rng
  float *A_h, *b_h, *c_h, *c_cpu;        // host arrays
  float *A_d, *b_d, *c_d;                // device arrays
  size_t mat_sz = N * N * sizeof(float); // matrix array size in bytes
  size_t vec_sz = N * sizeof(float);     // vector array size in bytes

  // allocate host memory
  A_h = (float*)malloc(mat_sz);
  b_h = (float*)malloc(vec_sz);
  c_h = (float*)malloc(vec_sz);
  c_cpu = (float*)malloc(vec_sz);

  // allocate device memory
  CUDA_CHECK(cudaMalloc(&A_d, mat_sz));
  CUDA_CHECK(cudaMalloc(&b_d, vec_sz));
  CUDA_CHECK(cudaMalloc(&c_d, vec_sz));

  // init input host arrays
  initMatRand(A_h, N);
  initVecRand(b_h, N);

  // copy input arrays from host to device
  CUDA_CHECK(cudaMemcpy(A_d, A_h, mat_sz, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b_h, vec_sz, cudaMemcpyHostToDevice));

  // launch matrix vector product kernel
  dim3 dimBlock(THREADS_PER_BLOCK, 1);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
  matVecProd<<<dimGrid, dimBlock>>>(A_d, b_d, c_d, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // copy output array from device to host
  CUDA_CHECK(cudaMemcpy(c_h, c_d, vec_sz, cudaMemcpyDeviceToHost));

  // compute matvec prod on cpu
  cpuMatVecProd(A_h, b_h, c_cpu, N);

  // verify result
  for (int i = 0; i < 5; ++i) {
    printf("c_h[%d] = %f, c_cpu[%d] = %f\n", i, c_h[i], i, c_cpu[i]);
  }

  // clean up memory
  free(A_h);
  free(b_h);
  free(c_h);
  free(c_cpu);
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(b_d));
  CUDA_CHECK(cudaFree(c_d));

  fprintf(stdout, "MATRIX-VECTOR PRODUCT PROGRAM COMPLETE.\n");
  return 0;
}

/* matrix-vector product */
__global__ void matVecProd(const float* A, const float* b, float* c, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n) {
    float acc = 0.0f;
    for (int col = 0; col < n; ++col) {
      acc += A[row * n + col] * b[col];
    }
    c[row] = acc;
  }
}

/* matrix vector product on the cpu */
void cpuMatVecProd(const float* A, const float* b, float* c, int n) {
  for (int i = 0; i < n; ++i) { // i-th row of A
    float acc = 0.0;
    for (int j = 0; j < n; ++j) {
      acc += A[i * n + j] * b[j];
    }
    c[i] = acc;
  }
}