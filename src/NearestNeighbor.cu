#include <cstdio>
#include "NearestNeighbor.h"

__global__ void computeErrorKernel(const double *U, const double *L,
                                   const double *candidate, double *errors,
                                   unsigned int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;

  double temp = candidate[i];
  double upper = U[i];
  double lower = L[i];

  // 避免分支，通过数学运算代替if-else
  errors[i] = max(0.0, temp - upper) + max(0.0, lower - temp);
}

__global__ void reduceKernel(double *input, double *output, unsigned int n) {
  __shared__ double sdata[NearestNeighbor::BLOCK_SZ];

  // 每个线程负责读取一个元素到共享内存
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    sdata[tid] = input[i];
  else
    sdata[tid] = 0;
  __syncthreads();

  // 进行并行规约，每一步都将活动线程数减半
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // 将每个block的规约结果写回全局内存
  if (tid == 0) atomicAdd(output, sdata[0]);
}