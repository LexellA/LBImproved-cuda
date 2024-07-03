#pragma once

#include "NearestNeighbor.h"

class LB_Keogh : public NearestNeighbor {
 public:
  LB_Keogh(double* v, unsigned int size, unsigned int constraint);
  ~LB_Keogh();

  double test(const double* candidate);
  double justlb(const double* candidate);
  double getLowestCost();
  int getNumberOfDTW() { return full_dtw; }

  int getNumberOfCandidates() { return lb_keogh; }


protected:
  int lb_keogh;
  int full_dtw;
  unsigned int size;
  double bestsofar;
  double *V_K;
  double* U_K;
  double* L_K;
  double *d_candidate, *d_errors, *d_result;

  cudaStream_t mStream;
  cudaGraph_t mGraph;
  cudaGraphExec_t mGraphExec;
};

__global__ void reduceKernel(double *input, double *output, unsigned int n);
__global__ void computeErrorKernel(const double *U, const double *L, const double *candidate, double *errors, unsigned int size);