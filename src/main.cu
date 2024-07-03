
#include "LB_Improved.h"
#include "dtw_origin.h"
#include "LB_Keogh.h"

#include <vector>
#include <iostream>
#include <chrono>


std::vector<double> getrandomwalk(uint size) {
  std::vector<double> data(size);
  data[0] = 0.0;
  for (uint k = 1; k < size; ++k)
    data[k] = (1.0 * rand() / (RAND_MAX)) - 0.5 + data[k - 1];
  return data;
}

void test_lbimproved(unsigned int arraySize, unsigned int numSeries, std::vector<double>& v, std::vector<std::vector<double>>& series) {
  double bestsofarGPU1 = 1e9;
  auto start = std::chrono::high_resolution_clock::now();
  LB_Improved lb(v.data(), arraySize, arraySize / 10);
  for (int i = 0; i < numSeries; i++) {
    double newbest = lb.test(series[i].data());
    if (newbest < bestsofarGPU1) {
      std::cout << "GPU(lbimproved) found a new nearest neighbor, distance (L1 "
                   "norm) = "
                << newbest << std::endl;
      bestsofarGPU1 = newbest;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "GPU(lbimproved) time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "ms" << std::endl;

  double bestsofarCPU1 = 1e9;
  start = std::chrono::high_resolution_clock::now();
  Origin::LB_Improved lb_origin(v, arraySize / 10);
  for (int i = 0; i < numSeries; i++) {
    double newbest = lb_origin.test(series[i]);
    if (newbest < bestsofarCPU1) {
      std::cout << "CPU(lbimproved) found a new nearest neighbor, distance (L1 "
                   "norm) = "
                << newbest << std::endl;
      bestsofarCPU1 = newbest;
    }
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "CPU(lbimproved) time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "ms" << std::endl;
}

void test_lbkeogh(unsigned int arraySize, unsigned int numSeries,
                  std::vector<double>& v,
                  std::vector<std::vector<double>>& series) {
  double bestsofarGPU2 = 1e9;
  auto start = std::chrono::high_resolution_clock::now();
  LB_Keogh lbk(v.data(), arraySize, arraySize / 10);
  for (int i = 0; i < numSeries; i++) {
    double newbest = lbk.test(series[i].data());
    if (newbest < bestsofarGPU2) {
      std::cout << "GPU(lbkeogh) found a new nearest neighbor, distance (L1 norm) = "
                << newbest << std::endl;
      bestsofarGPU2 = newbest;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "GPU(lbkeogh) time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "ms" << std::endl;


  double bestsofarCPU2 = 1e9;
  start = std::chrono::high_resolution_clock::now();
  Origin::LB_Keogh lbk_origin(v, arraySize / 10);
  for (int i = 0; i < numSeries; i++) {
    double newbest = lbk_origin.test(series[i]);
    if (newbest < bestsofarCPU2) {
      std::cout << "CPU(lbkeogh) found a new nearest neighbor, distance (L1 "
                   "norm) = "
                << newbest << std::endl;
      bestsofarCPU2 = newbest;
    }
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "CPU(lbkeogh) time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << "ms" << std::endl;
}

int main() {
  unsigned int arraySize = 20000;
  unsigned int numSeries = 500;
  std::vector<double> v = getrandomwalk(arraySize);
  std::vector<std::vector<double>> series(numSeries);
  for (int i = 0; i < numSeries; i++) {
    series[i] = getrandomwalk(arraySize);
  }

  test_lbimproved(arraySize, numSeries, v, series);
  test_lbkeogh(arraySize, numSeries, v, series);
  return 0;
}