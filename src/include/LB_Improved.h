#pragma once

#include <vector>
#include <string>
#include "NearestNeighbor.h"

class LB_Improved : public NearestNeighbor {
 public:
  LB_Improved(const std::vector<double> &v, int constraint);
  ~LB_Improved();

  void resetStatistics();
  double justlb(const std::vector<double> &candidate);
  double test(const std::vector<double> &candidate);
  std::string dumpTextDescriptor(const std::vector<double> &candidate);
  int getNumberOfDTW();
  int getNumberOfCandidates();
  double getLowestCost();

 private:
  int lb_keogh;
  int full_dtw;
  const std::vector<double> V;
  std::vector<double> buffer;
  int mConstraint;
  double bestsofar;
  std::vector<double> U;
  std::vector<double> L;
  std::vector<double> U2;
  std::vector<double> L2;
};