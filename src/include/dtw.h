#pragma once

#include <vector>
#include <iostream>
#define checkCudaErrors(call) \
do { \
    cudaError_t err = call; \
    if (cudaSuccess != err) { \
        std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << "." << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

class dtw {
 public:
  enum { INF = 1000000000 ,fast = true};
  dtw(unsigned int n, unsigned int constraint);
  ~dtw();
  double fastdynamic(const double* v, const double* w);
  double fastdynamic_1(const double* v, const double* w);
  double fastdynamic_origin(const std::vector<double> &v, const std::vector<double> &w);

 private:
  double* mGamma;
  std::vector<std::vector<double>> mGamma_origin;
  int mN;
  int mConstraint;
};