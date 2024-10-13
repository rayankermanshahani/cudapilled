/* simple vector addition */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

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

int main(int argc, char **argv) {
  fprintf(stdout, "PROGRAM COMPLETE\n");
  return 0;
}
