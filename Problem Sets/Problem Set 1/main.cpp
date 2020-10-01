// Udacity HW1 Solution

#include "compare.h"
#include "reference_calc.h"
#include "timer.h"
#include "utils.h"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <string>

void your_rgba_to_greyscale(const uchar4 *const h_rgbaImage,
                            uchar4 *const d_rgbaImage,
                            unsigned char *const d_greyImage, size_t numRows,
                            size_t numCols);

// include the definitions of the above functions for this homework
#include "HW1.cpp"

int main(int argc, char **argv) {
  uchar4 *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  std::string input_file;
  std::string output_file;
  std::string reference_file;
  double perPixelError = 0.0;
  double globalError = 0.0;
  bool useEpsCheck = false;
  switch (argc) {
  case 2:
    input_file = std::string(argv[1]);
    output_file = "HW1_output.png";
    reference_file = "HW1_reference.png";
    break;
  case 3:
    input_file = std::string(argv[1]);
    output_file = std::string(argv[2]);
    reference_file = "HW1_reference.png";
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
    std::cerr << "Usage: ./HW1 input_file [output_filename] "
                 "[reference_filename] [perPixelError] [globalError]"
              << std::endl;
    exit(1);
  }
  // load the image and give us our input and output pointers
  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage,
             input_file);

  size_t numPixels = numRows() * numCols();

  auto cpuStart = std::chrono::high_resolution_clock::now();

  // CPU implementation.
  for (int i = 0; i < numPixels; i++) {
    uchar4 *img = &h_rgbaImage[i];
    h_greyImage[i] = .299f * (*img).x + .587f * (*img).y + .114f * (*img).z;
  }

  auto cpuEnd = std::chrono::high_resolution_clock::now();
  auto cpuElapsedTime =
      std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart);

  int err;
  err = printf("CPU took: %f msecs.\n", cpuElapsedTime.count() / 1000000.);

  GpuTimer timer;
  timer.Start();
  // call the students' code
  your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(),
                         numCols());
  timer.Stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  err = printf("GPU took: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    // Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!"
              << std::endl;
    exit(1);
  }

  checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage,
                             sizeof(unsigned char) * numPixels,
                             cudaMemcpyDeviceToHost));

  // check results and output the grey image
  postProcess(output_file, h_greyImage);

  referenceCalculation(h_rgbaImage, h_greyImage, numRows(), numCols());

  postProcess(reference_file, h_greyImage);

  // generateReferenceImage(input_file, reference_file);
  compareImages(reference_file, output_file, useEpsCheck, perPixelError,
                globalError);

  cleanup();

  return 0;
}
