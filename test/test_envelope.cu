#include <chrono>
#include <deque>
#include <iostream>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "Envelope.h"
using namespace std;

void computeEnvelope(const vector<double> &array, unsigned int constraint,
                     vector<double> &maxvalues, vector<double> &minvalues);

__global__ void computeEnvelopeKernelThreadPerWindow(const double *array,
                                                     unsigned int size,
                                                     unsigned int constraint,
                                                     double *maxvalues,
                                                     double *minvalues);

__global__ void warmup() {

}

int main() {
  const int arraySize = 10000;

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dis(0, 1000);
  vector<double> array(arraySize);
  for (int i = 0; i < arraySize; i++) {
    array[i] = dis(gen);
  }

  double* d_array, *d_maxvalues, *d_minvalues;
  cudaMalloc(&d_array, arraySize * sizeof(double));
  cudaMalloc(&d_maxvalues, arraySize * sizeof(double));
  cudaMalloc(&d_minvalues, arraySize * sizeof(double));
  cudaMemcpy(d_array, array.data(), arraySize * sizeof(double),
             cudaMemcpyHostToDevice);


  warmup<<<1, 1>>>();

  //测试
  vector<double> maxvalues(array.size());
  vector<double> minvalues(array.size());
  auto start2 = chrono::high_resolution_clock::now();
  Envelope envelope(d_array, d_maxvalues, d_minvalues, arraySize,
                    arraySize / 10);
  envelope.compute();
  cudaDeviceSynchronize();
  auto end2 = chrono::high_resolution_clock::now();
  cout << "GPU Time: "
       << chrono::duration_cast<chrono::microseconds>(end2 - start2).count()
       << "us" << endl;

  cudaMemcpy(maxvalues.data(), d_maxvalues, arraySize * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(minvalues.data(), d_minvalues, arraySize * sizeof(double),
             cudaMemcpyDeviceToHost);


  vector<double> maxvalues2(array.size());
  vector<double> minvalues2(array.size());
  auto start3 = chrono::high_resolution_clock::now();
  computeEnvelope(array, arraySize / 10, maxvalues2, minvalues2);
  auto end3 = chrono::high_resolution_clock::now();
  cout << "CPU Time: "
       << chrono::duration_cast<chrono::microseconds>(end3 - start3).count()
       << "us" << endl;
  

  for (int i = 0; i < array.size(); i++) {
    if (maxvalues[i] != maxvalues2[i] || minvalues[i] != minvalues2[i]) {
      cout << "Mismatch at index " << i << endl;
      cout << "CPU: " << maxvalues2[i] << " " << minvalues2[i] << endl;
      cout << "GPU: " << maxvalues[i] << " " << minvalues[i] << endl;
      return 1;
    }
  }

  return 0;
}

typedef double floattype;
typedef unsigned int uint;

void computeEnvelope(const vector<floattype> &array, uint constraint,
                     vector<floattype> &maxvalues,
                     vector<floattype> &minvalues) {
  uint width = 1 + 2 * constraint;
  deque<int> maxfifo, minfifo;
  maxfifo.push_back(0);
  minfifo.push_back(0);
  for (uint i = 1; i < array.size(); ++i) {
    if (i >= constraint + 1) {
      maxvalues[i - constraint - 1] = array[maxfifo.front()];
      minvalues[i - constraint - 1] = array[minfifo.front()];
    }
    if (array[i] > array[i - 1]) {  // overshoot
      maxfifo.pop_back();
      while (maxfifo.size() > 0) {
        if (array[i] <= array[maxfifo.back()]) break;
        maxfifo.pop_back();
      }
    } else {
      minfifo.pop_back();
      while (minfifo.size() > 0) {
        if (array[i] >= array[minfifo.back()]) break;
        minfifo.pop_back();
      }
    }
    maxfifo.push_back(i);
    minfifo.push_back(i);
    if (i == width + maxfifo.front())
      maxfifo.pop_front();
    else if (i == width + minfifo.front())
      minfifo.pop_front();
  }
  for (uint i = array.size(); i <= array.size() + constraint; ++i) {
    if (i >= constraint + 1) {
      maxvalues[i - constraint - 1] = array[maxfifo.front()];
      minvalues[i - constraint - 1] = array[minfifo.front()];
    }
    if (i - maxfifo.front() >= width) maxfifo.pop_front();
    if (i - minfifo.front() >= width) minfifo.pop_front();
  }
}