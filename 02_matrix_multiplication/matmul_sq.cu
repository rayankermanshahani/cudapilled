#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024 // square matrix width
#define THREADS_PER_BLOCK 32

#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

/* function declarations */
__global__ void matMul(const float* A, const float* B, float* C, int n);
__global__ void matMulRow(const float* A, const float* B, float* C, int n);
__global__ void matMulCol(const float* A, const float* B, float* C, int n);
void cpuMatMul(const float* A, const float* B, float* C, int n);
void initMatRand(float* A, int n);
void cudaCheck(cudaError_t err, const char* file, int line);

/* driver program */
int main(void) {
  srand(time(NULL));                                  // initialize rng
  float *A_h, *B_h, *C_h, *C_h_row, *C_h_col, *C_cpu; // host arrays
  float *A_d, *B_d, *C_d;                             // device arrays
  size_t size = N * N * sizeof(float);                // array size in bytes

  // allocate host memory
  A_h = (float*) malloc(size);
  B_h = (float*) malloc(size);
  C_h = (float*) malloc(size);
  C_h_row = (float*) malloc(size);
  C_h_col = (float*) malloc(size);
  C_cpu = (float*) malloc(size);

  // allocate device memory
  CUDA_CHECK(cudaMalloc(&A_d, size));
  CUDA_CHECK(cudaMalloc(&B_d, size));
  CUDA_CHECK(cudaMalloc(&C_d, size));

  // init input host arrays
  initMatRand(A_h, N);
  initMatRand(B_h, N);

  // copy input arrays from host to device
  CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

  // launch first matmul kernel (thread -> output element)
  dim3 dimBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1);
  dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
               (N + dimBlock.y - 1) / dimBlock.y, 1);
  matMul<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // copy output array from device to host
  CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

  // launch second matmul kernel (thread -> output row)
  dim3 newDimBlock(THREADS_PER_BLOCK, 1, 1);
  dim3 newDimGrid((N + dimBlock.x - 1) / dimBlock.x, 1, 1);
  matMulRow<<<newDimGrid, newDimBlock>>>(A_d, B_d, C_d, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // copy output array from device to host
  CUDA_CHECK(cudaMemcpy(C_h_row, C_d, size, cudaMemcpyDeviceToHost));

  // launch third matmul kernel (thread -> output column)
  matMulCol<<<newDimGrid, newDimBlock>>>(A_d, B_d, C_d, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // copy output array from device to host
  CUDA_CHECK(cudaMemcpy(C_h_col, C_d, size, cudaMemcpyDeviceToHost));

  // compute matmul on cpu
  cpuMatMul(A_h, B_h, C_cpu, N);

  // verify result
  for (int i = 0; i < 5; ++i) {
    printf("C_gpu[%d] = %f, C_gpu_row[%d] = %f, C_gpu_row[%d] = %f, C_cpu[%d] "
           "= %f\n",
           i, C_h[i], i, C_h_row[i], i, C_h_col[i], i, C_cpu[i]);
  }

  // clean up memory
  free(A_h);
  free(B_h);
  free(C_h);
  free(C_h_row);
  free(C_h_col);
  free(C_cpu);
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

  fprintf(stdout, "MATRIX MULTIPLICATION PROGRAM COMPLETE.\n");
  return 0;
}

/* matrix multiplication kernel */
__global__ void matMul(const float* A, const float* B, float* C, int n) {
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

/* matrix multiplication: each thread corresponds to an output row */
__global__ void matMulRow(const float* A, const float* B, float* C, int n) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int k = 0; k < n; ++k) {
        acc += A[row * n + k] * B[k * n + col];
      }
      C[row * n + col] = acc;
    }
  }
}

/* matrix multiplication: each thread corresponds to an output column */
__global__ void matMulCol(const float* A, const float* B, float* C, int n) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < n) {
    for (int row = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int k = 0; k < n; ++k) {
        acc += A[row * n + k] * B[k * n + col];
      }
      C[row * n + col] = acc;
    }
  }
}

/* matrix multiplication on the cpu */
void cpuMatMul(const float* A, const float* B, float* C, int n) {
  for (int i = 0; i < n; ++i) {   // i-th row of A
    for (int j = 0; j < n; ++j) { // j-th col of B
      float acc = 0.0f;
      for (int k = 0; k < n; ++k) { // k-th element of corresponding row and col
        acc += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = acc;
    }
  }
}

/* randomly initalize float array for square matrix */
void initMatRand(float* a, int n) {
  for (int i = 0; i < n * n; ++i) { // square matrix
    a[i] = (float) rand() / (float) RAND_MAX;
  }
}

/* cuda error handling */
void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
