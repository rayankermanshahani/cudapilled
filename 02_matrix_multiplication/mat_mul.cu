/* simple vector addition */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

/* number of elements in each array (object) */
#define N 1000

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

/* driver program */
int main(void) {
  /* seed rng */
  srand(time(NULL));
  fprintf(stdout, "MATRIX MULTIPLICATION PROGRAM COMPLETE.\n");
  return 0;
}
/* matrix multiplication kernel */
__global__ void matmulKernel(const float *A, const float *B, float *C, int w) {
  /* row and col thread indices are the indices for the output element in C */
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < w) && (col < w)) {
    float acc = 0;
    for (int k = 0; k < w; ++k) {
      acc += A[row * w + k] * B[k * w + col]; /* A[row,k] * B[k,col] */
    }
    C[row * w + col] = acc; /* C[row,col] = ∑(A[row,k] * B[k,col]) */
  }
}

/* matmul between M (i x j) and N (j x k) produces P (i x k)
 *  - each element of P is an inner product of a row of M and a column of N
 *  - P_row,col denotes the element at row-th position in the vertical direction
 * and col-th position in the horizontal direction
 */
