#include "./include/dtw.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>

__global__ void init_mGamma(double *gamma, int size, double value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
      gamma[idx] = value;
    }
}

__global__ void compute_DTW(double*v ,double*w, double* mGamma, int wavefront, int constraint ,int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 一维grid和一维block

    // 限制i的范围
    if(i > wavefront || i >= N) return;

    // 超过constraint的不计算
    int j = wavefront - i;
    if(abs(i-j) > constraint  || j < 0 || j >= N) return;

    // printf("Thread [%d, %d]: i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);

    // mGamma[i][j] = fabs(v[i] - w[j]) + min(mGamma[i-1][j], mGamma[i][j-1], mGamma[i-1][j-1])
    double best = 1000000000;
    if (i > 0)
          best = mGamma[(i - 1) * N + j];
    if (j > 0)
      best = min(best, mGamma[i * N + j - 1]);
    if ((i > 0) && (j > 0))
      best = min(best, mGamma[(i - 1) * N + j - 1]);
    if ((i == 0) && (j == 0))
      mGamma[i * N + j] = fabs(v[i] - w[j]);
    else
      mGamma[i * N + j] = best + fabs(v[i] - w[j]);
    // printf("best = %f mGamma[%d][%d] = %f\n", best ,i, j, mGamma[i * N + j]);
  }


dtw::dtw(uint n, uint constraint)
      :mGamma_origin(n ,std::vector<double>(n ,INF)) ,mN(n), mConstraint(constraint) {
    
  int size = n*n;
  checkCudaErrors(cudaMallocManaged(&mGamma ,size*sizeof(double)));

  int K = 256;
  int NBlks = ceil((float)size / K); // 计算所需的块数
  init_mGamma<<<NBlks, K>>>(mGamma, size, std::numeric_limits<double>::infinity());

  // 等待GPU完成
  cudaDeviceSynchronize();
};

dtw::~dtw(){
  checkCudaErrors(cudaFree(mGamma));
  mGamma_origin.clear();
}

double dtw::fastdynamic(const std::vector<double>& v, const std::vector<double>& w) {
  if(!fast)
    return 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 记录开始时间
  cudaEventRecord(start);

  double* d_v;
  checkCudaErrors(cudaMalloc(&d_v, v.size() * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_v, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice));
  double* d_w;
  checkCudaErrors(cudaMalloc(&d_w, w.size() * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_w, w.data(), w.size() * sizeof(double), cudaMemcpyHostToDevice));

  int N = v.size();
  int K = 512; // 每个块的线程数
  int NBlks = ceil((float)N / K); // 计算所需的块数
  // 波前从0开始一直到2n  
  for(int warefront = 0 ;warefront <= 2 * (N - 1) ;warefront++){
    compute_DTW<<<NBlks,K>>>(d_v ,d_w ,mGamma ,warefront ,mConstraint ,mN);
    std::cout << std::endl << "warefront = " << warefront << std::endl;
    checkCudaErrors(cudaDeviceSynchronize());
    // for(int i = 0 ;i < N*N ;i++){
    //   std::cout << mGamma[i] << " ";
    // }
  }

  // 记录结束时间
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // 计算并打印执行时间
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "CUDA Execution time: " << milliseconds << " ms " << "fd: " << mGamma[(mN - 1) * N + mN - 1] << std::endl;

  // 清理
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCudaErrors(cudaFree(d_v));
  checkCudaErrors(cudaFree(d_w));
  return mGamma[(mN - 1) * N + mN - 1];
}

double dtw::fastdynamic_origin(const std::vector<double>& v, const std::vector<double>& w){
  if (!fast)
    return 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 记录开始时间
  cudaEventRecord(start);

  assert(static_cast<int>(v.size()) == mN);
  assert(static_cast<int>(w.size()) == mN);
  assert(static_cast<int>(mGamma_origin.size()) == mN);
  double Best(INF);

  for (int i = 0; i < mN; ++i) {
    assert(static_cast<int>(mGamma_origin[i].size()) == mN);
    // 这里直接把constraint限制在了i的范围内，这样可以减少计算量
    for (int j = std::max(0, i - mConstraint);
         j < std::min(mN, i + mConstraint + 1); ++j) {
      // printf("i = %d, j = %d\n",i, j);
      // 边获取边比较
      Best = INF;
      if (i > 0)
        Best = mGamma_origin[i - 1][j];
      if (j > 0) Best = std::min(Best, mGamma_origin[i][j - 1]);
      if ((i > 0) && (j > 0))
        Best = std::min(Best, mGamma_origin[i - 1][j - 1]);
      if ((i == 0) && (j == 0))
        mGamma_origin[i][j] = fabs(v[i] - w[j]);
      else
        mGamma_origin[i][j] = Best + fabs(v[i] - w[j]);
      // printf("best = %f mGamma[%d][%d] = %f\n", Best ,i, j, mGamma[i][j]);
    }
    // printf("\n");
  }
  // 记录结束时间
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // 计算并打印执行时间
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Origin Execution time: " << milliseconds << " ms " << "fd: " << mGamma_origin[mN - 1][mN - 1] << std::endl;

  // 清理
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return mGamma_origin[mN - 1][mN - 1];
}