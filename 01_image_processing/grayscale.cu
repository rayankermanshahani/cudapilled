// grayscale.cu -- process a color image into grayscale

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

#define IMG_SIZE 1500 * 2000 /* 1500 x 2000 pixels */
#define CHANNELS 3           /* 3 channels corresponding to RGB */

/* function declarations */
int loadJPGImage(const char *filename, int *width, int *height, int *channels,
                 unsigned char *P);
int saveJPGImage(const char *filename, int width, int height, int channels,
                 const unsigned char *P);
void cudaCheck(cudaError_t err, const char *file, int line);
__global__ void grayscaleKernel(unsigned char *Pout, unsigned char *Pin,
                                int width, int height);

int main(int argc, char **argv) {
  int y = 450;
  int x = 400;
  int channels = 3;

  const char *file_in = "./imgs/pic.jpg";
  const char *file_out = "./imgs/g_pic.jpg";

  /* size of image in 1D array for flattened image in bytes */
  unsigned int size = IMG_SIZE * sizeof(unsigned char);

  /* declare host arrays */
  unsigned char *Pin_h, *Pout_h;

  /* declare device arrays */
  unsigned char *Pin_d, *Pout_d;

  /* allocate memory for host arrays */
  Pin_h = (unsigned char *)malloc(size);
  Pout_h = (unsigned char *)malloc(size);

  /* allocate memory for device arrays */
  CUDA_CHECK(cudaMalloc((void **)&Pin_d, size));
  CUDA_CHECK(cudaMalloc((void **)&Pout_d, size));

  /* load JPG image into input host array */
  if (!loadJPGImage(file_in, &x, &y, &channels, Pin_h)) {
    fprintf(stderr, "loadJPGImage() error:\n");
    exit(EXIT_FAILURE);
  }

  /* copy input array from host to device */
  CUDA_CHECK(cudaMemcpy(Pin_d, Pin_h, IMG_SIZE, cudaMemcpyHostToDevice));

  /* launch config parameters */
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(ceil(x / dimBlock.x), ceil(y / dimBlock.y), 1); /* 25, 29, 1 */

  /* launch kernel */
  grayscaleKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, x, y);
  CUDA_CHECK(cudaDeviceSynchronize());

  /* copy output array from device to host */
  CUDA_CHECK(cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost));

  if (!saveJPGImage(file_out, x, y, channels, Pout_h)) {
    fprintf(stderr, "saveJPGImage() error:\n");
  }

  /* free host memory */
  free(Pin_h);
  free(Pout_h);

  /* free device memory */
  CUDA_CHECK(cudaFree(Pin_d));
  CUDA_CHECK(cudaFree(Pout_d));

  fprintf(stdout, "\n");
  fprintf(stdout, "PROGRAM COMPLETE\n");
  return 0;
}

/* load jpg image from specified filepath into array */
int loadJPGImage(const char *filename, int *width, int *height, int *channels,
                 unsigned char *P) {
  /* load jpg image via stbi */
  unsigned char *data = stbi_load(filename, width, height, channels, 0);

  /* error handling */
  if (data == NULL) {
    fprintf(stderr, "Failed to load image from %s\n", filename);
    return 0;
  }

  /* copy image bytes to array */
  size_t size = (*width) * (*height) * (*channels);
  mempcpy(P, data, size);
  stbi_image_free(data);

  return 1;
}

/* save array as jpg image at specified filepath */
int saveJPGImage(const char *filename, int width, int height, int channels,
                 const unsigned char *P) {
  int result = stbi_write_jpg(filename, width, height, channels, P, 100);
  if (result == 0) {
    fprintf(stderr, "Failed to save image to %s\n", filename);
    return 0;
  }
  return 1;
}

/* cuda error handling */
void cudaCheck(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA ERROR] at file %s:%d\n%s\n", file, line,
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/* convert an rgb image to grayscale */
__global__ void grayscaleKernel(unsigned char *Pout, unsigned char *Pin,
                                int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    /* get 1D offset for the grayscale image */
    int grayOffset = row * width + col;
    /* get RGB offsets for the grayscale image */
    int rgbOffset = grayOffset * CHANNELS;
    unsigned char r = Pin[rgbOffset];     /* red value */
    unsigned char g = Pin[rgbOffset + 1]; /* green value */
    unsigned char b = Pin[rgbOffset + 2]; /* blue value */

    /* compute luminance from constants */
    Pout[grayOffset] = 0.21f * r + 0.72f * g + 0.07f * b;
  }
}