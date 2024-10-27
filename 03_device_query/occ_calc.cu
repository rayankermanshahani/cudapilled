#include <cuda_occupancy.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

/* function declarations */
void cudaCheck(cudaError_t err, const char *file, int line);
void calculateOccupancy(const void *kernel);

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
    fprintf(stdout, "  warp size: %d\n", deviceProperties.warpSize);
    fprintf(stdout, "  clock rate: %d kHz\n", deviceProperties.clockRate);
    fprintf(stdout, "  max threads per block: %d\n",
            deviceProperties.maxThreadsPerBlock);
    fprintf(stdout, "  total number of SMs: %d\n",
            deviceProperties.multiProcessorCount);
    fprintf(stdout, "  max threads per SM: %d\n",
            deviceProperties.maxThreadsPerMultiProcessor);
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
void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stdout, "%s in %s at line %d\n", cudaGetErrorString(err), file,
            line);
    exit(EXIT_FAILURE);
  }
}

/* calculate the SM occupancy of a kernel */
void calculateOccupancy(const void *kernel) {
  // kernel attributes
  cudaFuncAttributes attribs;
  CUDA_CHECK(cudaFuncGetAttributes(&attribs, kernel));

  // device properties
  cudaDeviceProp props;
  int deviceId;
  CUDA_CHECK(cudaGetDevice(&deviceId));
  CUDA_CHECK(cudaGetDeviceProperties_v2(&props, deviceId));

  // variables for occupancy calculation
  int blockSize;   // selected block size
  int minGridSize; // min grid size for max occupancy
  int numBlocks;   // desired grid size

  // calculate optimal block size
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &blockSize, kernel,
      0, // dynamic shared memory bytes if needed
      0  // block size limit if needed
      ));

  // calculate theoretical occupancy
  int maxActiveBlocks;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, kernel, blockSize,
      0 // dynamic shared memory size
      ));

  float occupancy = (maxActiveBlocks * blockSize) /
                    float(props.warpSize * props.maxThreadsPerMultiProcessor);

  fprintf(stdout, "kernel properties:\n");
  fprintf(stdout, "  register count: %d\n", attribs.numRegs);
  fprintf(stdout, "  constant memory: %zu bytes\n", attribs.constSizeBytes);
  fprintf(stdout, "  local memory: %zu bytes\n", attribs.localSizeBytes);
  fprintf(stdout, "  max threads per block: %d\n", attribs.maxThreadsPerBlock);
  fprintf(stdout, "\noccupancy info:\n");
  fprintf(stdout, "  suggested block size: %d\n", blockSize);
  fprintf(stdout, "  theoretical occupancy: %.2f%%\n", occupancy * 100);
}
