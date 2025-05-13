#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Error checking macro */
#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

/* CUDA error handling function */
inline void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA ERROR] at file %s:%d\n%s\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/* Initialize a vector with random float values */
inline void initRand(float* a, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = (float)rand() / (float)RAND_MAX;
  }
}
/* Initialize a vector with random float values (alias for consistency) */
inline void initVecRand(float* a, int n) { initRand(a, n); }

/* Initialize a square matrix with random float values */
inline void initMatRand(float* A, int n) {
  for (int i = 0; i < n * n; ++i) {
    A[i] = (float)rand() / (float)RAND_MAX;
  }
}

#endif