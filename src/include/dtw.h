#pragma once

#include <vector>

class dtw {
 public:
  enum { INF = 1000000000 };
  dtw(unsigned int n, unsigned int constraint);
  double fastdynamic(const std::vector<double>& v, const std::vector<double>& w);
 private:
  std::vector<std::vector<double>> mGamma;
  int mN;
  int mConstraint;
};