#include <cuda_runtime.h>

#include "NearestNeighbor.h"
#include "dtw.h"
#include "LB_Improved.h"
#include "Envelope.h"


__global__ void computeErrorKernelBuffer(const double *U, const double *L,
                                         const double *candidate,
                                         double *errors, double *buffer,
                                         unsigned int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        double error = 0, r_buffer = 0;
        double temp = candidate[i];
        double upper = U[i];
        double lower = L[i];
        if (temp > upper)
        {
            error = temp - upper;
            r_buffer = upper;
        }
        else if (temp < lower)
        {
            error = lower - temp;
            r_buffer = lower;
        }
        else
        {
          error = 0;
            r_buffer = temp;
        }
        errors[i] = error;
        buffer[i] = r_buffer;
    }
}



/**
 * 规约计算数组和
 */
double LB_Improved::compute_sum(double *d_array, uint size) {
  int threadsPerBlock = BLOCK_SZ;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;


  cudaMemsetAsync(d_result, 0, sizeof(double), mStream);
  reduceKernel<<<blocksPerGrid, threadsPerBlock,0, mStream>>>(
      d_array, d_result, size);
  // 从GPU复制最终结果回CPU
  double h_result;
  cudaMemcpyAsync(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost,
                  mStream);
  
  cudaStreamEndCapture(mStream, &mGraph);
  cudaGraphInstantiate(&mGraphExec, mGraph, NULL, NULL, 0);
  cudaGraphLaunch(mGraphExec, mStream);
  cudaStreamSynchronize(mStream);
  cudaGraphExecDestroy(mGraphExec);
  cudaGraphDestroy(mGraph);
  return h_result;
}

double LB_Improved::test(const double *candidate)
{
    ++lb_keogh;
    // 将数据从CPU复制到GPU
    cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal);
    
    cudaMemcpyAsync(d_candidate, candidate, size * sizeof(double),
                    cudaMemcpyHostToDevice, mStream);
    

    // 调用computeErrorKernel，计算出每个点的误差errors
    int threadsPerBlock = LB_Improved::BLOCK_SZ;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    computeErrorKernelBuffer<<<blocksPerGrid, threadsPerBlock, 0, mStream>>>(
        U_K, L_K, d_candidate, d_errors, d_buffer, size);

    double error = compute_sum(d_errors, size);

    

    // Continue with the rest of the test function
    if (error < bestsofar) {

      // 第二次LB_Keogh
        cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal);
        Envelope envelope(d_buffer, d_U2, d_L2, size, mConstraint);
        envelope.compute(mStream);

        computeErrorKernel<<<blocksPerGrid, threadsPerBlock, 0, mStream>>>(d_U2, d_L2, V_K, d_errors, size);

        double error2 = compute_sum(d_errors, size);

        error2 += error;

        if (error2 < bestsofar)
        {
          ++full_dtw;
          cudaStreamBeginCapture(mStream, cudaStreamCaptureModeGlobal);
          const double trueerror = mDTW.fastdynamic(V_K, d_candidate, mStream, mGraph, mGraphExec);

          if (trueerror < bestsofar) bestsofar = trueerror;
        }
    }

    cudaStreamSynchronize(mStream);

    return bestsofar;
}

double LB_Improved::getLowestCost() { return bestsofar; }

LB_Improved::LB_Improved(double *v, unsigned int v_size,
                         unsigned int constraint)
    : NearestNeighbor(v, v_size, constraint),
      size(v_size),
      lb_keogh(0),
      full_dtw(0),
      bestsofar(dtw::INF),
      mConstraint(constraint),
      mGraphExec(NULL)
      {

    cudaStreamCreate(&mStream);

    cudaMallocAsync(&V_K, size * sizeof(double), mStream);
    cudaMallocAsync(&U_K, size * sizeof(double), mStream);
    cudaMallocAsync(&L_K, size * sizeof(double), mStream);

    cudaMallocAsync(&d_candidate, size * sizeof(double), mStream);
    cudaMallocAsync(&d_errors, size * sizeof(double), mStream);
    cudaMallocAsync(&d_buffer, size * sizeof(double), mStream);
    cudaMallocAsync(&d_U2, size * sizeof(double), mStream);
    cudaMallocAsync(&d_L2, size * sizeof(double), mStream);

    cudaMallocAsync(&d_result, sizeof(double), mStream);

    cudaMemcpyAsync(V_K, v, size * sizeof(double), cudaMemcpyHostToDevice,
                    mStream);

    Envelope envelope(V_K, U_K, L_K, size, mConstraint);
    envelope.compute(mStream);
    cudaStreamSynchronize(mStream);
}

LB_Improved::~LB_Improved()
{
  cudaFreeAsync(V_K, mStream);
  cudaFreeAsync(U_K, mStream);
  cudaFreeAsync(L_K, mStream);

  cudaFreeAsync(d_candidate, mStream);
  cudaFreeAsync(d_errors, mStream);
  cudaFreeAsync(d_buffer, mStream);

  cudaFreeAsync(d_U2, mStream);
  cudaFreeAsync(d_L2, mStream);

  cudaFreeAsync(d_result, mStream);
  cudaStreamDestroy(mStream);
}