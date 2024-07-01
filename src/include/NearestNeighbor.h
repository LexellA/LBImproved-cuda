#pragma once

#include <vector>
#include "dtw.h"

class NearestNeighbor {
 public:
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
