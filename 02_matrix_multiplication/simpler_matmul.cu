/* matmul_square.cu -- matrix mulitplication assuming square input matrices  */

#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define N 1024 /* square matrix width */
#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

/* function declarations */
void cudaCheck(cudaError_t err, const char *file, int line);
void initMatrix(float *A, int n);
__global__ void matmulKernel(const float *A, const float *B, float *C, int n);

/* driver function */
int main(int argc, char **argv) {
  float *A_h, *B_h, *C_h; // host arrays
  float *A_d, *B_d, *C_d; // device arrays

  fprintf(stdout, "MATRIX MULTIPLICATION PROGRAM COMPLETE\n");

  return 0;
}

/* cuda error handling */
void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA ERROR] at file %s:%d\n%s\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/* randonly initialize square matrix */
void initMatrix(float *A, int n) {
  for (int i = 0; i < n * n; ++i) {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

/* matrix multiplication: assuming that operand matrices are squares */
__global__ void matmulKernel(const float *A, const float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y; /* row in A */
  int col = blockIdx.x * blockDim.x + threadIdx.x; /* col in B */

  /* check if row and col values are valid */
  if ((row < n) && (col < n)) {
    float acc = 0.0f;
    /* compute inner dot product of A's row and B's col */
    for (int k = 0; k < n; ++k) {
      acc += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = acc;
  }
}
