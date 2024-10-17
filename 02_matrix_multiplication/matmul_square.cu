/* matmul_square.cu -- matrix mulitplication assuming square input matrices  */

#include <cmath>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

/* function declarations */
void cudaCheck(cudaError_t err, const char *file, int line);
void initMatrix(float *A, int n);
bool compareMatrices(const float *A, const float *B, int n, float tolerance);
bool verifyMatmulKernel(void (*kernel)(const float *, const float *, float *,
                                       int),
                        int n, float tolerance);
__global__ void matmulKernel(const float *A, const float *B, float *C, int n);

/* driver function */
int main(int argc, char **argv) {
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
  for (int i = 0; i < n; ++i) {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

/* compare matrices with a tolerance */
bool compareMatrices(const float *A, const float *B, int n, float tolerance) {
  for (int i = 0; i < n; ++i) {
    if (std::fabs(A[i] - B[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

/* verify cuda matmul kernel executed correctly */
bool verifyMatmulKernel(void (*kernel)(const float *, const float *, float *,
                                       int),
                        int n, float tolerance = 1e-5) {
  size_t size = n * n * sizeof(float);

  /* declare host matrices */
  float *A_h, *B_h, *C_h_cuda, *C_h_cublas;

  /* declare device matrices */
  float *A_d, *B_d, *C_d;

  /* allocate memory for host matrices */
  A_h = new float[n * n];
  B_h = new float[n * n];
  C_h_cuda = new float[n * n];
  C_h_cublas = new float[n * n];

  /* allocate memory for device matrices */
  CUDA_CHECK(cudaMalloc((void **)&A_d, size));
  CUDA_CHECK(cudaMalloc((void **)&B_d, size));
  CUDA_CHECK(cudaMalloc((void **)&C_d, size));

  /* initalize host input matrices */
  initMatrix(A_h, n);
  initMatrix(B_h, n);

  /* copy host input matrices to device */
  CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

  /* launch matmul kernel */
  kernel(A_d, B_d, C_d, n);

  /* copy result device matrix to host */
  CUDA_CHECK(cudaMemcpy(C_h_cuda, C_d, size, cudaMemcpyDeviceToHost));

  /* compute reference result via cuBLAS */
  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  float alpha = 1.0f;
  float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B_d, n, A_d, n,
              &beta, C_d, n);
  CUDA_CHECK(cudaMemcpy(C_h_cublas, C_d, size, cudaMemcpyDeviceToHost));
  cublasDestroy_v2(handle);

  /* compare results */
  bool correct = compareMatrices(C_h_cuda, C_h_cublas, n, tolerance);
  if (correct) {
    fprintf(stdout, "Success: CUDA kernel output matches cuBLAS result within "
                    "specified tolerance.\n");
  } else {
    fprintf(stdout,
            "Failure: CUDA kernel output does not match cuBLAS result.\n");
  }

  /* clean up memory */
  delete[] A_h;
  delete[] B_h;
  delete[] C_h_cuda;
  delete[] C_h_cublas;
  CUDA_CHECK(cudaFree(A_d));
  CUDA_CHECK(cudaFree(B_d));
  CUDA_CHECK(cudaFree(C_d));

  return correct;
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
