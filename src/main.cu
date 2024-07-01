#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include "dtw.h"

std::vector<double> get_rand_seq(uint size) {
  std::vector<double> data(size);
  data[0] = 0.0;
  for (uint k = 1; k < size; ++k) {
    data[k] = (1.0 * rand() / (RAND_MAX)) - 0.5 + data[k - 1];
    // std::cout << data[k] << " ";
  }
  return data;
}

__global__ void computeDtw(double* v, double* w, double* mGamma, unsigned int N, unsigned int constraint) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = gid; i < N * N; i += stride) {
    mGamma[i] = dtw::INF;
  }

  if (gid == 0) {
    mGamma[0] = fabs(v[0] - w[0]);
  }

  __syncthreads();


  for (int k = 1; k < N; k++) {
    for (int id = gid; id <= k; id += stride) {
      int i = k - id;
      int j = id;

      if (abs(i - j) > constraint)
        continue;
      if (i - 1 < 0) {
        mGamma[i * N + j] = mGamma[i * N + j - 1] + fabs(v[i] - w[j]);
      } else if (j - 1 < 0) {
        mGamma[i * N + j] = mGamma[(i - 1) * N + j] + fabs(v[i] - w[j]);
      } else {
        mGamma[i * N + j] = fabs(v[i] - w[j]) + min(min(mGamma[(i - 1) * N + j],
                                                        mGamma[i * N + j - 1]),
                                                    mGamma[(i - 1) * N + j - 1]);        
      }
    }
    __syncthreads();
  }

  for (int k = 1; k < N; k++) {
    for (int id = gid; id < N - k; id += stride) {
      int i = N - 1 - id;
      int j = k + id;

      if (abs(i - j) > constraint)
        continue;

      mGamma[i * N + j] = fabs(v[i] - w[j]) + min(min(mGamma[(i - 1) * N + j],
                                                      mGamma[i * N + j - 1]),
                                                  mGamma[(i - 1) * N + j - 1]);
    }
    __syncthreads();
  }
}

double fastdynamic_origin(const std::vector<double>& v,
                          const std::vector<double>& w, int mN, int mConstraint,
                          std::vector<std::vector<double>>& mGamma_origin) {

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // 记录开始时间
  checkCudaErrors(cudaEventRecord(start));

  double Best(100000000);

  for (int i = 0; i < mN; ++i) {
    for (int j = std::max(0, i - mConstraint);
         j < std::min(mN, i + mConstraint + 1); ++j) {
      // printf("i = %d, j = %d\n",i, j);
      // 边获取边比较
      Best = 100000000;
      if (i > 0) Best = mGamma_origin[i - 1][j];
      if (j > 0) Best = std::min(Best, mGamma_origin[i][j - 1]);
      if ((i > 0) && (j > 0))
        Best = std::min(Best, mGamma_origin[i - 1][j - 1]);
      if ((i == 0) && (j == 0))
        mGamma_origin[i][j] = fabs(v[i] - w[j]);
      else
        mGamma_origin[i][j] = Best + fabs(v[i] - w[j]);
    }
  }
  return mGamma_origin[mN - 1][mN - 1];
}

int main() {
  int array_size = 10000;
  std::vector<double> v = get_rand_seq(array_size);
  std::vector<double> w = get_rand_seq(array_size);

  double *d_v, *d_w, *d_mGamma;
  cudaMalloc(&d_v, array_size * sizeof(double));
  cudaMalloc(&d_w, array_size * sizeof(double));
  cudaMalloc(&d_mGamma, array_size * array_size * sizeof(double));
  cudaMemcpy(d_v, v.data(), array_size * sizeof(double),
              cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, w.data(), array_size * sizeof(double),
              cudaMemcpyHostToDevice);

  int blockSize = 128;
  int numBlocks = (array_size + blockSize - 1) / blockSize;

  auto start = std::chrono::high_resolution_clock::now();
  computeDtw<<<numBlocks, blockSize>>>(d_v, d_w, d_mGamma, array_size,
                                       array_size / 10);
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  std::vector<double> mGamma(array_size * array_size);
  cudaMemcpy(mGamma.data(), d_mGamma,
              array_size * array_size * sizeof(double),
              cudaMemcpyDeviceToHost);

  std::cout << "GPU Time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us" << std::endl;


  std::vector<std::vector<double>> mGamma_origin(
      array_size, std::vector<double>(array_size, 100000000));

  start = std::chrono::high_resolution_clock::now();
  double ans =
      fastdynamic_origin(v, w, array_size, array_size / 10, mGamma_origin);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "CPU Time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us" << std::endl;


  for(int i = 0; i < 10; i++){
    for (int j = 0; j < 10; j++) {
      std::cout << mGamma[i * array_size + j] << " ";
    }
    std::cout << std::endl;
  }

  for(int i = 0; i < 10; i++){
    for (int j = 0; j < 10; j++) {
      std::cout << mGamma_origin[i][j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}