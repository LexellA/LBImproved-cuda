#pragma once

#include <iostream>
#include <vector>

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
  double fastdynamic(double* v, double* w);
  double fastdynamic(double* v, double* w, cudaStream_t& stream, cudaGraph_t& graph, cudaGraphExec_t& graphExec);
  double fastdynamic_SC(double* v, double* w);

 private:
  double* mGamma;
  std::vector<std::vector<double>> mGamma_origin;
  int mN;
  int mConstraint;
};