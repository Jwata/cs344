/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <cstddef>

__global__ void max_sequential(float *maxLuminance, const size_t sizeLum) {
  for (size_t i = 1; i < sizeLum; i++) {
    maxLuminance[0] = max(maxLuminance[i], maxLuminance[0]);
  }
}

__global__ void min_sequential(float *minLuminance, const size_t sizeLum) {
  for (size_t i = 1; i < sizeLum; i++) {
    minLuminance[0] = min(minLuminance[i], minLuminance[0]);
  }
}

__global__ void max_reduce(const float *const in, float *out, const int size) {
  __shared__ float s_data[1024];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Copy global memory to shared memory in parallel.
  if (gid < size)
    s_data[tid] = in[gid];
  __syncthreads_count(blockDim.x);

  // Parallel reduction within the block.
  for (int n = blockDim.x / 2; n > 0; n >>= 1) {
    if (tid < n)
      s_data[tid] = max(s_data[tid], s_data[tid + n]);
    __syncthreads_count(blockDim.x);
  }
  if (tid == 0)
    out[blockIdx.x] = s_data[0];

  // Note: it turned out that this implementation is slower than CPU.
  // Only thread 0 does the following steps.
  // if (tid == 0) {
  //   float max_value;
  //   max_value = s_data[0];
  //   for (int i = 1; i < blockDim.x; i++) {
  //     max_value = max(max_value, s_data[i]);
  //   }
  //   // printf("block:%d, value:%f\n", blockIdx.x, max_value);
  //   out[blockIdx.x] = max_value;
  // }
}

__global__ void min_reduce(const float *const in, float *out, const int size) {
  __shared__ float s_data[1024];

  const int tid = threadIdx.x;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Copy global memory to shared memory in parallel.
  if (gid < size)
    s_data[tid] = in[gid];
  __syncthreads_count(blockDim.x);

  // Parallel reduction within the block.
  for (int n = blockDim.x / 2; n > 0; n >>= 1) {
    if (tid < n)
      s_data[tid] = min(s_data[tid], s_data[tid + n]);
    __syncthreads_count(blockDim.x);
  }
  if (tid == 0)
    out[blockIdx.x] = s_data[0];

  // Note: it turned out that this implementation is slower than CPU.
  // Only thread 0 does the following steps.
  // if (tid == 0) {
  //   float min_value;
  //   min_value = s_data[0];
  //   for (int i = 1; i < blockDim.x; i++) {
  //     min_value = min(min_value, s_data[i]);
  //   }
  //   // printf("block:%d, value:%f\n", blockIdx.x, min_value);
  //   out[blockIdx.x] = min_value;
  // }
}

__global__ void genHistogram(unsigned int *d_hist, const float *const in,
                             size_t size, size_t d_numBins, float *d_maxValue,
                             float *d_minValue) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Calculate range.
  float valueMax = d_maxValue[0];
  float valueMin = d_minValue[0];
  float valueRange = valueMax - valueMin;
  unsigned int numBins = d_numBins;
  float binRange = valueRange / numBins;

  // printf("valueMax=%f, valueMin=%f, valueRange=%f, numBins=%d,
  // binRange=%f\n",
  //        valueMax, valueMin, valueRange, numBins, binRange);

  int bin = (in[gid] - valueMin) / binRange;

  // TODO: sum up within shared memory before updating global memory.
  atomicAdd(&d_hist[bin], 1);
}

void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf, float &min_logLum,
                                  float &max_logLum, const size_t numRows,
                                  const size_t numCols, const size_t numBins) {
  // TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  // TODO: Reduction can be more efficient.
  // See the reference for more details.
  // Optimizing Parallel Reduction in CUDA
  // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  size_t sizeAcc = sizeof(float) * numCols * numRows / 1024;
  float *d_maxLuminance;
  float *d_minLuminance;
  checkCudaErrors(cudaMalloc(&d_maxLuminance, sizeAcc));
  checkCudaErrors(cudaMalloc(&d_minLuminance, sizeAcc));

  int numBlocks = ceil(numCols * numRows / 1024.);
  max_reduce<<<numBlocks, 1024>>>(d_logLuminance, d_maxLuminance,
                                  numRows * numCols);
  max_reduce<<<1, 1024>>>(d_maxLuminance, d_maxLuminance, 1024);
  min_reduce<<<numBlocks, 1024>>>(d_logLuminance, d_minLuminance,
                                  numRows * numCols);
  min_reduce<<<1, 1024>>>(d_minLuminance, d_minLuminance, 1024);
  cudaDeviceSynchronize();

  checkCudaErrors(cudaMemcpy(&max_logLum, d_maxLuminance, sizeof(float),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&min_logLum, d_minLuminance, sizeof(float),
                             cudaMemcpyDeviceToHost));
  printf("maxLum %f, minLum %f\n", max_logLum, min_logLum);

  unsigned int *d_histogram;
  size_t sizeHist = sizeof(int) * numBins;
  checkCudaErrors(cudaMalloc(&d_histogram, sizeHist));
  genHistogram<<<numBlocks, 1024>>>(d_histogram, d_logLuminance,
                                    numRows * numCols, numBins, d_maxLuminance,
                                    d_minLuminance);
  cudaDeviceSynchronize();

  // unsigned int h_histogram[numBins];
  // checkCudaErrors(
  //     cudaMemcpy(&h_histogram, d_histogram, sizeHist,
  //     cudaMemcpyDeviceToHost));
}
