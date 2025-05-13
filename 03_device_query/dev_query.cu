#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

/* function declarations */
void cudaCheck(cudaError_t err, const char* file, int line);

/* driver program */
int main(void) {
  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  fprintf(stdout, "%d CUDA devices detected on this machine\n", deviceCount);

  cudaDeviceProp deviceProperties;
  for (int i = 0; i < deviceCount; ++i) {
    fprintf(stdout, "\nCUDA Device %d:\n", i);
    cudaGetDeviceProperties_v2(&deviceProperties, i);
    // do stuff with device properties
    cudaUUID_t uuid = deviceProperties.uuid;
    fprintf(stdout, "  UUID: %s\n", uuid.bytes);
    fprintf(stdout, "  name: %s\n", deviceProperties.name);
    fprintf(stdout, "  major compute capability: %d\n", deviceProperties.major);
    fprintf(stdout, "  minor compute capability: %d\n", deviceProperties.minor);
    fprintf(stdout, "  total global memory: %lf GB\n",
            (double)deviceProperties.totalGlobalMem / (1024 * 1024 * 1024));
    fprintf(stdout, "  total constant memory: %lf GB\n",
            (double)deviceProperties.totalConstMem / (1024 * 1024 * 1024));
    fprintf(stdout, "  warp size: %d\n", deviceProperties.warpSize);
    fprintf(stdout, "  clock rate: %d kHz\n", deviceProperties.clockRate);
    fprintf(stdout, "  max threads per block: %d\n",
            deviceProperties.maxThreadsPerBlock);
    fprintf(stdout, "  max registers per block: %d\n",
            deviceProperties.regsPerBlock);
    fprintf(stdout, "  total number of SMs: %d\n",
            deviceProperties.multiProcessorCount);
    fprintf(stdout, "  max threads per SM: %d\n",
            deviceProperties.maxThreadsPerMultiProcessor);
    fprintf(stdout, "  max registers per SM: %d\n",
            deviceProperties.regsPerMultiprocessor);
    fprintf(stdout, "  dimension z:\n    max blocks: %d\n    max threads: %d\n",
            deviceProperties.maxGridSize[2], deviceProperties.maxThreadsDim[2]);
    fprintf(stdout, "  dimension y:\n    max blocks: %d\n    max threads: %d\n",
            deviceProperties.maxGridSize[1], deviceProperties.maxThreadsDim[1]);
    fprintf(stdout, "  dimension x:\n    max blocks: %d\n    max threads: %d\n",
            deviceProperties.maxGridSize[0], deviceProperties.maxThreadsDim[0]);
  }

  fprintf(stdout, "\nDEVICE QUERY PROGRAM COMPLETE.\n");
  return 0;
}

/* cuda error handling */
void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    fprintf(stdout, "%s in %s at line %d\n", cudaGetErrorString(err), file,
            line);
    exit(EXIT_FAILURE);
  }
}
