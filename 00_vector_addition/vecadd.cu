#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000 // number of array elements
#define THREADS_PER_BLOCK 256

#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

/* function declarations */
__global__ void vecAdd(const float* a, const float* b, float* c, int n);
__global__ void vecAddAdj(const float* a, const float* b, float* c, int n);
__global__ void vecAddSec(const float* a, const float* b, float* c, int n);
void initRand(float* a, int n);
void cudaCheck(cudaError_t err, const char* file, int line);

/* driver program */
int main(void) {
  srand(time(NULL));                          // initialize rng
  float *a_h, *b_h, *c_h, *c_h_adj, *c_h_sec; // host arrays
  float *a_d, *b_d, *c_d;                     // device arrays
  size_t size = N * sizeof(float);            // array size in bytes

  // allocate host memory
  a_h = (float*)malloc(size);
  b_h = (float*)malloc(size);
  c_h = (float*)malloc(size);
  c_h_adj = (float*)malloc(size);
  c_h_sec = (float*)malloc(size);

  // allocate device memory
  CUDA_CHECK(cudaMalloc(&a_d, size));
  CUDA_CHECK(cudaMalloc(&b_d, size));
  CUDA_CHECK(cudaMalloc(&c_d, size));

  // init input host arrays
  initRand(a_h, N);
  initRand(b_h, N);

  // copy input arrays from host to device
  CUDA_CHECK(cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice));

  // launch kernel for simple vector addition
  dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
  dim3 blocksPerGrid(ceil(N / float(threadsPerBlock.x)), 1, 1);
  vecAdd<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  // copy output array from device to host
  CUDA_CHECK(cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost));

  // launch kernel that processes adjacent array elements
  vecAddAdj<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c_h_adj, c_d, size, cudaMemcpyDeviceToHost));

  // launch kernel that processes adjacent array sections
  vecAddSec<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c_h_sec, c_d, size, cudaMemcpyDeviceToHost));

  // verify result
  for (int i = 0; i < 5; ++i) {
    printf(
        "a[%d] = %f, b[%d] = %f, c[%d] = %f, c_adj[%d] = %f, c_sec[%d] = %f\n",
        i, a_h[i], i, b_h[i], i, c_h[i], i, c_h_adj[i], i, c_h_sec[i]);
  }

  // clean up memory
  free(a_h);
  free(b_h);
  free(c_h);
  free(c_h_adj);
  free(c_h_sec);
  CUDA_CHECK(cudaFree(a_d));
  CUDA_CHECK(cudaFree(b_d));
  CUDA_CHECK(cudaFree(c_d));

  fprintf(stdout, "VECTOR ADDITION PROGRAM COMPLETE.\n");
  return 0;
}

/* vector addition: each thread processes one output element */
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

/* vector addition kernel: each thread processes two adjacent array elements */
__global__ void vecAddAdj(const float* a, const float* b, float* c, int n) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;

  if (i < n) { // first adjacent element
    c[i] = a[i] + b[i];
  }

  if (i + 1 < n) { // second adjacent element
    c[i + 1] = a[i + 1] + b[i + 1];
  }
}

/* vector addition: each thread processes two sections of the input arrays */
__global__ void vecAddSec(const float* a, const float* b, float* c, int n) {
  unsigned i = (blockDim.x * blockIdx.x * 2) +
               threadIdx.x; // each section is separated by blockDim.x elements

  if (i < n) { // first section
    c[i] = a[i] + b[i];
  }

  unsigned j = i + blockDim.x; // second section
  if (j < n) {
    c[j] = a[j] + b[j];
  }
}

/* randomly initalize float array */
void initRand(float* a, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = (float)rand() / (float)RAND_MAX;
  }
}

/* cuda error handling */
void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
