// Udacity HW2 Driver

#include "timer.h"
#include "utils.h"
#include <cstddef>
#include <iostream>
#include <stdio.h>
#include <string>

#include "compare.h"
#include "reference_calc.h"

// include the definitions of the above functions for this homework
#include "HW2.cpp"

/*******  DEFINED IN student_func.cu *********/

void your_gaussian_blur(const uchar4 *const h_inputImageRGBA,
                        uchar4 *const d_inputImageRGBA,
                        uchar4 *const d_outputImageRGBA, const size_t numRows,
                        const size_t numCols, unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred, const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage,
                                const size_t numColsImage,
                                const float *const h_filter,
                                const size_t filterWidth);

void gaussian_blur_with_cpu(const uchar4 *const inputImageRGBA,
                            uchar4 *const outputImageRGBA, const size_t numRows,
                            const size_t numCols, float *filter,
                            const size_t filterWidth) {

  for (int i = 0; i < numRows * numCols; i++) {
    const int x = i / numCols;
    const int y = i % numCols;
    const auto filterBaseX = x - (filterWidth - 1) / 2;
    const auto filterBaseY = y - (filterWidth - 1) / 2;
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;
    for (int f_i = 0; f_i < filterWidth * filterWidth; f_i++) {
      const int f_x = filterBaseX + f_i / filterWidth;
      const int f_y = filterBaseY + f_i % filterWidth;
      if (f_x < 0 || f_x >= numRows || f_y < 0 || f_y >= numCols)
        continue;
      const auto pixel_pos = f_x * numCols + f_y;
      const auto pixel = inputImageRGBA[pixel_pos];
      r += static_cast<float>(pixel.x) * filter[f_i];
      g += static_cast<float>(pixel.y) * filter[f_i];
      b += static_cast<float>(pixel.z) * filter[f_i];
    }
    outputImageRGBA[i].x = r;
    outputImageRGBA[i].y = g;
    outputImageRGBA[i].z = b;
    outputImageRGBA[i].w = 255;
  }
}

/*******  Begin main *********/

int main(int argc, char **argv) {
  int err;
  uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

  float *h_filter;
  int filterWidth;

  std::string input_file;
  std::string output_file;
  std::string reference_file;
  double perPixelError = 0.0;
  double globalError = 0.0;
  bool useEpsCheck = false;
  switch (argc) {
  case 2:
    input_file = std::string(argv[1]);
    output_file = "HW2_output.png";
    reference_file = "HW2_reference.png";
    break;
  case 3:
    input_file = std::string(argv[1]);
    output_file = std::string(argv[2]);
    reference_file = "HW2_reference.png";
    break;
  case 4:
    input_file = std::string(argv[1]);
    output_file = std::string(argv[2]);
    reference_file = std::string(argv[3]);
    break;
  case 6:
    useEpsCheck = true;
    input_file = std::string(argv[1]);
    output_file = std::string(argv[2]);
    reference_file = std::string(argv[3]);
    perPixelError = atof(argv[4]);
    globalError = atof(argv[5]);
    break;
  default:
    std::cerr << "Usage: ./HW2 input_file [output_filename] "
                 "[reference_filename] [perPixelError] [globalError]"
              << std::endl;
    exit(1);
  }
  // load the image and give us our input and output pointers
  preProcess(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA,
             &d_outputImageRGBA, &d_redBlurred, &d_greenBlurred, &d_blueBlurred,
             &h_filter, &filterWidth, input_file);

  // Run CPU implementaion.
  auto cpuStart = std::chrono::high_resolution_clock::now();
  gaussian_blur_with_cpu(h_inputImageRGBA, h_outputImageRGBA, numRows(),
                         numCols(), h_filter, filterWidth);
  auto cpuEnd = std::chrono::high_resolution_clock::now();
  auto cpuElapsedTime =
      std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart);
  err = printf("CPU took: %f msecs.\n", cpuElapsedTime.count() / 1000000.);

  allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);
  GpuTimer timer;
  timer.Start();
  // call the students' code
  your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA,
                     numRows(), numCols(), d_redBlurred, d_greenBlurred,
                     d_blueBlurred, filterWidth);
  timer.Stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
  err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    // Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!"
              << std::endl;
    exit(1);
  }

  // check results and output the blurred image

  size_t numPixels = numRows() * numCols();
  // copy the output back to the host
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__,
                             sizeof(uchar4) * numPixels,
                             cudaMemcpyDeviceToHost));

  postProcess(output_file, h_outputImageRGBA);

  // referenceCalculation(h_inputImageRGBA, h_outputImageRGBA, numRows(),
  //                      numCols(), h_filter, filterWidth);

  //  Cheater easy way with OpenCV
  generateReferenceImage(input_file, reference_file, filterWidth);
  postProcess(reference_file, h_outputImageRGBA);

  compareImages(reference_file, output_file, useEpsCheck, perPixelError,
                globalError);

  checkCudaErrors(cudaFree(d_redBlurred));
  checkCudaErrors(cudaFree(d_greenBlurred));
  checkCudaErrors(cudaFree(d_blueBlurred));

  cleanUp();

  return 0;
}
