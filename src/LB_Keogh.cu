#include <cuda_runtime.h>

#include "NearestNeighbor.h"
#include "dtw.h"
#include "LB_Keogh.h" 
#include "Envelope.h"



double LB_Keogh::test(const double* candidate)
{
    ++lb_keogh;

    cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal);

    // 将数据从CPU复制到GPU
    cudaMemcpyAsync(d_candidate, candidate, size * sizeof(double),
               cudaMemcpyHostToDevice, mStream);

    // 调用computeErrorKernel，计算出每个点的误差errors
    int threadsPerBlock = BLOCK_SZ;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    computeErrorKernel<<<blocksPerGrid, threadsPerBlock, 0, mStream>>>(
        U_K, L_K, d_candidate, d_errors, size);

    // 调用reduceKernel
    cudaMemsetAsync(d_result, 0, sizeof(double), mStream);
    reduceKernel<<<blocksPerGrid, threadsPerBlock, 0, mStream>>>(
        d_errors, d_result, size);

    // 从GPU复制最终结果回CPU
    double error = 0;
    cudaMemcpyAsync(&error, d_result, sizeof(double), cudaMemcpyDeviceToHost,
                    mStream);

    cudaStreamEndCapture(mStream, &mGraph);
    cudaGraphInstantiate(&mGraphExec, mGraph, NULL, NULL, 0);
    cudaGraphLaunch(mGraphExec, mStream);
    cudaStreamSynchronize(mStream);
    cudaGraphDestroy(mGraph);
    cudaGraphExecDestroy(mGraphExec);

    // Continue with the rest of the test function
    if (error < bestsofar)
    {
      ++full_dtw;
      cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal);
      const double trueerror =
          mDTW.fastdynamic(V_K, d_candidate, mStream, mGraph, mGraphExec);
      if (trueerror < bestsofar) bestsofar = trueerror;
    }

    cudaStreamSynchronize(mStream); 

    return bestsofar;
}


double LB_Keogh::getLowestCost() { return bestsofar; }

LB_Keogh::LB_Keogh(double* v, unsigned int v_size, unsigned int constraint)
    : NearestNeighbor(v, v_size, constraint),
      size(v_size),
      lb_keogh(0),
      full_dtw(0),
      bestsofar(dtw::INF),
      mGraphExec(NULL){

    cudaStreamCreate(&mStream);

    cudaMallocAsync(&V_K, size * sizeof(double), mStream);
    cudaMallocAsync(&U_K, size * sizeof(double), mStream);
    cudaMallocAsync(&L_K, size * sizeof(double), mStream);
    cudaMallocAsync(&d_candidate, size * sizeof(double), mStream);
    cudaMallocAsync(&d_errors, size * sizeof(double), mStream);
    cudaMallocAsync(&d_result, sizeof(double), mStream);

    cudaMemcpyAsync(V_K, v, size * sizeof(double), cudaMemcpyHostToDevice,
                   mStream);

    Envelope envelope(V_K, U_K, L_K, size, constraint);
    envelope.compute(mStream);
}

LB_Keogh::~LB_Keogh()
{
  cudaFreeAsync(V_K, mStream);
  cudaFreeAsync(U_K, mStream);
  cudaFreeAsync(L_K, mStream);

  cudaFreeAsync(d_candidate, mStream);
  cudaFreeAsync(d_errors, mStream);
  cudaFreeAsync(d_result, mStream);
  cudaStreamDestroy(mStream);
}