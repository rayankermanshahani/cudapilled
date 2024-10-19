#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000 // number of array elements
#define THREADS_PER_BLOCK 256

/* function declarations */
__global__ void vecAdd(const float *a, const float *b, float *c, int n);
__global__ void vecAddAdj(const float *a, const float *b, float *c, int n);
__global__ void vecAddSec(const float *a, const float *b, float *c, int n);
void initRand(float *a, int n);
void checkCudaError(cudaError_t err, const char *file, int line);

#define cudaCheckError(err) checkCudaError(err, __FILE__, __LINE__);

/* driver program */
int main(void) {
  srand(time(NULL));               // initialize rng
  float *a_h, *b_h, *c_h;          // host arrays
  float *a_d, *b_d, *c_d;          // device arrays
  size_t size = N * sizeof(float); // array size in bytes

  // allocate host memory
  a_h = (float *)malloc(size);
  b_h = (float *)malloc(size);
  c_h = (float *)malloc(size);

  // allocate device memory
  cudaCheckError(cudaMalloc(&a_d, size));
  cudaCheckError(cudaMalloc(&b_d, size));
  cudaCheckError(cudaMalloc(&c_d, size));

  // init input host arrays
  initRand(a_h, N);
  initRand(b_h, N);

  // copy input arrays from host to device
  cudaCheckError(cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice));

  // launch kernel
  dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);
  dim3 blocksPerGrid(ceil(N / float(threadsPerBlock.x)), 1, 1);
  vecAdd<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, N);
  cudaCheckError(cudaDeviceSynchronize());

  // copy output array from device to host
  cudaCheckError(cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost));

  // verify result
  for (int i = 0; i < 5; ++i) {
    printf("a[%d] = %f, b[%d] = %f, c[%d] = %f\n", i, a_h[i], i, b_h[i], i,
           c_h[i]);
  }

  // clean up memory
  free(a_h);
  free(b_h);
  free(c_h);
  cudaCheckError(cudaFree(a_d));
  cudaCheckError(cudaFree(b_d));
  cudaCheckError(cudaFree(c_d));

  fprintf(stdout, "VECTOR ADDITION PROGRAM COMPLETE.\n");
  return 0;
}

/* vector addition: each thread processes one output element */
__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

/* vector addition kernel: each thread processes two adjacent array elements */
__global__ void vecAddAdj(const float *a, const float *b, float *c, int n) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;

  if (i < n) { // first adjacent element
    c[i] = a[i] + b[i];
  }

  if (i + 1 < n) { // second adjacent element
    c[i + 1] = a[i + 1] + b[i + 1];
  }
}

/* vector addition: each thread processes two sections of the input arrays */
__global__ void vecAddSec(const float *a, const float *b, float *c, int n) {
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
void initRand(float *a, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = (float)rand() / (float)RAND_MAX;
  }
}

/* cuda error handling */
void checkCudaError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
