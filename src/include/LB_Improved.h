#pragma once

#include "NearestNeighbor.h"

class LB_Improved : public NearestNeighbor
{
 public:
  LB_Improved(double *v, unsigned int size, unsigned int constraint);
  ~LB_Improved();

  double test(const double *candidate);
  double justlb(const double *candidate);
  double getLowestCost();
  int getNumberOfDTW() { return full_dtw; }
  double compute_sum(double *d_array, uint size);
  int getNumberOfCandidates() { return lb_keogh; }
  

protected:
  int lb_keogh;
  int full_dtw;
  unsigned int size;
  int mConstraint;
  double bestsofar;
  double *V_K;
  double *U_K;
  double *L_K;
  double *d_candidate, *d_errors, *d_buffer;
  double *d_U2, *d_L2;
  double *d_result;
};

__global__ void computeErrorKernelBuffer(const double *U, const double *L,
                                         const double *candidate,
                                         double *errors, double *buffer,
                                         unsigned int size);

