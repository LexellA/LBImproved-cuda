#pragma once

#include <vector>
#include <cassert>

#include "NearestNeighbor.h"

class LB_Improved : public NearestNeighbor
{
public:
  LB_Improved(double *v, unsigned int size, unsigned int constraint);
  ~LB_Improved();

  double test(const double *candidate);
  double test_kernel(const double *candidate);
  double justlb(const double *candidate);
  double getLowestCost();
  int getNumberOfDTW() { return full_dtw; }

  int getNumberOfCandidates() { return lb_keogh; }

protected:
  int lb_keogh;
  int full_dtw;
  unsigned int size;
  int mConstraint;
  double bestsofar;
  double *V;
  double *U;
  double *L;
  double *V_K;
  double *U_K;
  double *L_K;
};