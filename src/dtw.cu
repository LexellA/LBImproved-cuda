#include "./include/dtw.h"
#include <vector>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>

__global__ void init_mGamma(double *gamma, int size, double value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
      gamma[idx] = value;
    }
}

__global__ void compute_DTW(const double*v ,const double*w, double* mGamma, int wavefront, int constraint ,int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 一维grid和一维block

    // 限制i的范围
    if(i > wavefront || i >= N) return;

    // 超过constraint的不计算
    int j = wavefront - i;
    if(abs(i-j) > constraint  || j < 0 || j >= N) return;

    // 使用共享内存
    __shared__ double shared_v[512];
    __shared__ double shared_w[512];
    
    shared_v[threadIdx.x] = v[i];
    shared_w[threadIdx.x] = w[j];
    // __syncthreads();

    // printf("Thread [%d, %d]: i = %d, j = %d\n", blockIdx.x, threadIdx.x, i, j);

    // 递推公式：mGamma[i][j] = fabs(v[i] - w[j]) + min(mGamma[i-1][j], mGamma[i][j-1], mGamma[i-1][j-1])
    double best = 1000000000;
    if (i > 0)
      best = mGamma[(i - 1) * N + j];
    if (j > 0)
      best = min(best, mGamma[i * N + j - 1]);
    if ((i > 0) && (j > 0))
      best = min(best, mGamma[(i - 1) * N + j - 1]);
    if ((i == 0) && (j == 0))
      mGamma[i * N + j] = fabs(shared_v[threadIdx.x] - shared_w[threadIdx.x]);
    else
      mGamma[i * N + j] = best + fabs(shared_v[threadIdx.x] - shared_w[threadIdx.x]);
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
  checkCudaErrors(cudaDeviceSynchronize());
};

dtw::~dtw(){
  checkCudaErrors(cudaFree(mGamma));
  mGamma_origin.clear();
}

double dtw::fastdynamic(const double* v, const double* w) {
  if(!fast)
    return 0;

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // 记录开始时间
  checkCudaErrors(cudaEventRecord(start));

  int N = mN;
  int K = 32; // 每个块的线程数
  // 波前从0开始一直到2n  
  for(int wavefront = 0 ;wavefront <= 2 * (N - 1) ;wavefront++){
    int NBlks = ceil((float)(wavefront+1) / K); // 计算所需的块数
    compute_DTW<<<NBlks,K>>>(v ,w ,mGamma ,wavefront ,mConstraint ,mN);
    // std::cout << std::endl << "warefront = " << warefront << std::endl;
    // checkCudaErrors(cudaDeviceSynchronize());
    // for(int i = 0 ;i < N*N ;i++){
    //   std::cout << mGamma[i] << " ";
    // }
  }

  // 记录结束时间
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  // 计算并打印执行时间
  float milliseconds = 0;
  checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "CUDA Execution time: " << milliseconds << " ms " << "fd: " << mGamma[(mN - 1) * N + mN - 1] << std::endl;

  // 清理
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return mGamma[(mN - 1) * N + mN - 1];
}

// --------------------------------------------------

double dtw::fastdynamic_origin(const std::vector<double>& v, const std::vector<double>& w){
  auto start = std::chrono::high_resolution_clock::now(); // 开始计时

  if (!fast)
    return 0;


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

  auto end = std::chrono::high_resolution_clock::now(); // 结束计时
  std::chrono::duration<double, std::milli> elapsed = end - start; // 计算经过的时间（毫秒）

  std::cout << "Function execution took " << elapsed.count() << " milliseconds." << " fd: " << mGamma_origin[mN - 1][mN - 1] << std::endl;

  return mGamma_origin[mN - 1][mN - 1];
}

// ------------------------------------------
__global__ void compute_DTW_1(const double*v ,const double*w, double* mGamma, int constraint ,int N){
    extern __shared__ double shared_mem[];
    double* shared_v = shared_mem;
    double* shared_w = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int i = bid * blockDim.x + tid;
    if (i >= N) return;

    if (tid < N) {
        shared_v[tid] = v[tid];
        shared_w[tid] = w[tid];
    }

    __syncthreads();

    for (int wavefront = 0; wavefront <= 2 * (N - 1); wavefront++) {
        int j = wavefront - i;

        if (i <= wavefront && j >= 0 && j < N && abs(i - j) <= constraint) {
            double best = INFINITY;
            if (i > 0)
                best = mGamma[(i - 1) * N + j];
            if (j > 0)
                best = fmin(best, mGamma[i * N + j - 1]);
            if (i > 0 && j > 0)
                best = fmin(best, mGamma[(i - 1) * N + j - 1]);
            if (i == 0 && j == 0)
                mGamma[i * N + j] = fabs(shared_v[i] - shared_w[j]);
            else
                mGamma[i * N + j] = best + fabs(shared_v[i] - shared_w[j]);
        }
        __syncthreads();
    }
  }

double dtw::fastdynamic_1(const double* v, const double* w) {
  if(!fast)
        return 0;

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    int N = mN;
    int K = 128; // 每个块的线程数

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // 波前从0开始一直到2n  
    for(int wavefront = 0 ;wavefront <= 2 * (N - 1) ;wavefront++){
        int NBlks = ceil((float)(wavefront+1) / K); // 计算所需的块数
        compute_DTW<<<max(NBlks ,32), K, 2 * K * sizeof(double), stream>>>(v, w, mGamma, wavefront, mConstraint, mN);
    }

    checkCudaErrors(cudaStreamEndCapture(stream, &graph));
    checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    // 记录开始时间
    checkCudaErrors(cudaEventRecord(start));

    // 执行捕获的流
    checkCudaErrors(cudaGraphLaunch(instance, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // 记录结束时间
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    // 计算并打印执行时间
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "CUDA Execution time: " << milliseconds << " ms " << "fd: " << mGamma[(mN - 1) * N + mN - 1] << std::endl;

    // 清理
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaGraphExecDestroy(instance));

    return mGamma[(mN - 1) * N + mN - 1];
}