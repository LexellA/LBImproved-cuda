#include <cmath>
#include <vector>

#include "dtw.h"

__global__ void init_mGamma(double *gamma, int size, double value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
      gamma[idx] = value;
    }
}

dtw::dtw(uint n, uint constraint)
      :mGamma_origin(n ,std::vector<double>(n ,dtw::INF)) ,mN(n), mConstraint(constraint) {
  
  // 初始化mGamma为INF
  int size = mN * mN;
  cudaMalloc(&mGamma, size * sizeof(double));
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

// ----------------------------------------------------------------------

__global__ void compute_DTW(const double* v, const double* w, double* mGamma,
                            int wavefront, int constraint, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 一维grid和一维block

  // 限制i的范围
  if(i > wavefront || i >= N) return;

  // 超过constraint的不计算
  int j = wavefront - i;
  if(abs(i-j) > constraint  || j < 0 || j >= N) return;

  // 递推公式：mGamma[i][j] = fabs(v[i] - w[j]) + min(mGamma[i-1][j], mGamma[i][j-1], mGamma[i-1][j-1])
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
}


double dtw::fastdynamic(double* v, double* w) {
  if(!fast)
    return 0;

  int N = mN;
  int K = 32; // 每个块的线程数
  // 波前从0开始一直到2n  
  for(int wavefront = 0 ;wavefront <= 2 * (N - 1) ;wavefront++){
    int NBlks = ceil((float)(wavefront+1) / K); // 计算所需的块数
    compute_DTW<<<NBlks,K>>>(v ,w ,mGamma ,wavefront ,mConstraint ,mN);
  }

  double result;
  checkCudaErrors(cudaMemcpy(&result, mGamma + (mN - 1) * mN + mN - 1,
                             sizeof(double), cudaMemcpyDeviceToHost));
  return result;
}

// ----------------------------------加上流捕获------------------------------------------------ //
double dtw::fastdynamic_SC(double* v, double* w) {
    if (!fast)
        return 0;

    int N = mN;
    int K = 32; // 每个块的线程数

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    for (int wavefront = 0; wavefront <= 2 * (N - 1); wavefront++) {
        int NBlks = ceil((float)(wavefront + 1) / K); // 计算所需的块数
        compute_DTW<<<NBlks, K, 0, stream>>>(v, w, mGamma, wavefront, mConstraint, mN);
    }

    checkCudaErrors(cudaStreamEndCapture(stream, &graph));
    checkCudaErrors(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    // 执行捕获的流
    checkCudaErrors(cudaGraphLaunch(instance, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // // 输出结果
    // std::vector<double> gamma_cpu(mN * mN);
    // cudaMemcpy(gamma_cpu.data(), mGamma, mN * mN * sizeof(double), cudaMemcpyDeviceToHost);
    double result;
    checkCudaErrors(cudaMemcpy(&result, mGamma + (mN - 1) * mN + mN - 1,
                               sizeof(double), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaStreamDestroy(stream));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaGraphExecDestroy(instance));


    return result;
}

