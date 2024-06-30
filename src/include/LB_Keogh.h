#pragma once

#include <vector>

#include "NearestNeighbor.h"

class LB_Keogh : public NearestNeighbor {
 public:
  LB_Keogh(const std::vector<double>& v, unsigned int constraint);
  double test(const std::vector<double>& candidate);
  double justlb(const std::vector<double>& candidate);
  double getLowestCost();
  int getNumberOfDTW();
  int getNumberOfCandidates();
  void resetStatistics();
  ~LB_Keogh() {}

 protected:
  int lb_keogh;
  int full_dtw;
  const std::vector<double> V;
  int mConstraint;
  double bestsofar;
  std::vector<double> U;
  std::vector<double> L;
};