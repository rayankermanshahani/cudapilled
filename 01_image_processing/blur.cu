// grayscale.cu -- process a color image into grayscale

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(err) (cudaCheck(err, __FILE__, __LINE__))

#define BLUR_SIZE 3 /* dimension of square patch for blurring */
#define CHANNELS 3  /* RGB channels */

/* function declarations */
int loadJPGImage(const char *filename, int *width, int *height, int *channels,
                 unsigned char *P);
int saveJPGImage(const char *filename, int width, int height, int channels,
                 const unsigned char *P);
void cudaCheck(cudaError_t err, const char *file, int line);
__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h);

/* driver function */
int main(int argc, char **argv) {
  int width = 400;
  int height = 450;
  int channels = 3;

  const char *file_in = "./imgs/pic.jpg";
  const char *file_out = "./imgs/b_pic.jpg";

  /* bytes in 1D array of (flattened) image */
  unsigned int size = (width * height * channels) * sizeof(unsigned char);

  /* declare host ararys */
  unsigned char *Pin_h, *Pout_h;

  /* declare device ararys */
  unsigned char *Pin_d, *Pout_d;

  /* allocate memory for host arrays */
  Pin_h = (unsigned char *)malloc(size);
  Pout_h = (unsigned char *)malloc(size);

  /* allocate memory for host arrays */
  CUDA_CHECK(cudaMalloc((void **)&Pin_d, size));
  CUDA_CHECK(cudaMalloc((void **)&Pout_d, size));

  /* load JPG image into input host array */
  if (!loadJPGImage(file_in, &width, &height, &channels, Pin_h)) {
    fprintf(stderr, "loadJPGImage() error:\n");
    exit(EXIT_FAILURE);
  }

  /* copy input array from host to device */
  CUDA_CHECK(cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice));

  /* launch config parameters */
  dim3 dimBlock(16, 16, 1);
  dim3 dimGrid(ceil(width / (float)dimBlock.x),
               ceil(height / (float)dimBlock.y), 1);

  /* launch kernel */
  blurKernel<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  /* copy output array from device to host */
  CUDA_CHECK(cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost));

  /* save output host array into JPG image */
  if (!saveJPGImage(file_out, width, height, channels, Pout_h)) {
    fprintf(stderr, "saveJPGImage() error:\n");
  }

  /* free host memory */
  free(Pin_h);
  free(Pout_h);

  /* free device memory */
  CUDA_CHECK(cudaFree(Pin_d));
  CUDA_CHECK(cudaFree(Pout_d));

  fprintf(stdout, "BLUR PROGRAM COMPLETE\n");
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

/* blur an image */
__global__ void blurKernel(unsigned char *Pin, unsigned char *Pout, int w,
                           int h) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < h && col < w) {
    int pixValR = 0, pixValG = 0, pixValB = 0;
    int pixels = 0;
    /* get average of surrounding BLUR_SIZE x BLUR_SIZE patch */
    for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
      for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
        int curRow = row + blurRow;
        int curCol = col + blurCol;

        // verify we have a valid image pixel
        if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
          int idx = (curRow * w + curCol) * CHANNELS;
          pixValR += Pin[idx];
          pixValG += Pin[idx + 1];
          pixValB += Pin[idx + 2];
          ++pixels;
        }
      }
    }
    /* write new pixel to the output image */
    int outIdx = (row * w + col) * CHANNELS;
    Pout[outIdx] = (unsigned char)(pixValR / pixels);
    Pout[outIdx + 1] = (unsigned char)(pixValG / pixels);
    Pout[outIdx + 2] = (unsigned char)(pixValB / pixels);
  }
}
