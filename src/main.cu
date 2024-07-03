
#include "LB_Improved.h"
#include "Envelope.h"
#include "dtw_origin.h"

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

int main() {
  unsigned int arraySize = 10000;
  unsigned int numSeries = 50000;
  std::vector<double> v = getrandomwalk(arraySize);
  std::vector<std::vector<double>> series(numSeries);
  for (int i = 0; i < numSeries; i++) {
    series[i] = getrandomwalk(arraySize);
  }


  double bestsofarGPU1 = 1e9;
  auto start = std::chrono::high_resolution_clock::now();
  LB_Improved lb(v.data(), arraySize, arraySize / 10);
  for (int i = 0; i < numSeries; i++) {
    double newbest = lb.test(series[i].data());
    if (newbest < bestsofarGPU1) {
      std::cout << "GPU found a new nearest neighbor, distance (L1 norm) = "
                << newbest << std::endl;
      bestsofarGPU1 = newbest;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "GPU time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                   .count()
            << "ms" << std::endl;


  double bestsofarCPU1 = 1e9;
  start = std::chrono::high_resolution_clock::now();
  Origin::LB_Improved lb_origin(v, arraySize / 10);
  for (int i = 0; i < numSeries; i++) {
    double newbest = lb_origin.test(series[i]);
    if (newbest < bestsofarCPU1) {
      std::cout << "CPU found a new nearest neighbor, distance (L1 norm) = "
                << newbest << std::endl;
      bestsofarCPU1 = newbest;
    }
  }
  end = std::chrono::high_resolution_clock::now();
  std::cout << "CPU time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                   .count()
            << "ms" << std::endl;

  return 0;
}