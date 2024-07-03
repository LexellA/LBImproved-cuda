#pragma once

#include "dtw.h"

class NearestNeighbor {
 public:
  enum { BLOCK_SZ = 256 };
  
  NearestNeighbor(double* v,unsigned int size, unsigned int constraint)
    : mDTW(size, constraint) {}

  virtual double test(const double* candidate) = 0;
  virtual double getLowestCost() = 0;
  virtual ~NearestNeighbor() {}
  virtual int getNumberOfDTW() = 0;
  virtual int getNumberOfCandidates() = 0;

 protected:
  dtw mDTW;
};

__global__ void reduceKernel(double *input, double *output, unsigned int n);

__global__ void computeErrorKernel(const double *U, const double *L,
                                   const double *candidate, double *errors,
                                   unsigned int size);