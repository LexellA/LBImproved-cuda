#pragma once

#include <vector>
#include "dtw.h"

class NearestNeighbor {
 public:
  NearestNeighbor(const std::vector<double>& v, unsigned int constraint);
  virtual double test(const std::vector<double>& candidate) = 0;
  virtual double getLowestCost() = 0;
  virtual ~NearestNeighbor() {}
  virtual int getNumberOfDTW() = 0;
  virtual int getNumberOfCandidates() = 0;

 protected:
  dtw mDTW;
};
