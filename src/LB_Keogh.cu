#include <cuda_runtime.h>
#include <vector>

#include "dtw.h"
#include "LB_Keogh.h" 
#include "Envelope.h"

__global__ void computeErrorKernel(const double *U, const double *L, const double *candidate, double *errors, unsigned int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        if (candidate[i] > U[i])
        {
            errors[i] = candidate[i] - U[i];
        }
        else if (candidate[i] < L[i])
        {
            errors[i] = L[i] - candidate[i];
        }
        else
        {
            errors[i] = 0.0;
        }
    }
}

__global__ void reduceKernel(double *input, double *output, unsigned int n)
{
    extern __shared__ double sdata[];

    // 每个线程负责读取一个元素到共享内存
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        sdata[tid] = input[i];
    else
        sdata[tid] = 0;
    __syncthreads();

    // 进行并行规约，每一步都将活动线程数减半
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 将每个block的规约结果写回全局内存
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

double LB_Keogh::test_kernel(double* candidate)
{
    ++lb_keogh;

    // 分配device内存
    double  *d_U, *d_L, *d_candidate, *d_errors, *d_result;
    // cudaMalloc(&d_V, size * sizeof(double));
    cudaMalloc(&d_candidate, size * sizeof(double));
    cudaMalloc(&d_errors, size * sizeof(double));

    // 将数据从CPU复制到GPU
    cudaMemcpy(d_candidate, candidate, size * sizeof(double), cudaMemcpyHostToDevice);

    // 调用computeErrorKernel，计算出每个点的误差errors
    int threadsPerBlock = 512;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    computeErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(U_K, L_K, d_candidate, d_errors, size);

    /**
     * 规约计算errors
     */ 

    // 分配device内存用于存储每个block的规约结果
    cudaMalloc(&d_result, blocksPerGrid * sizeof(double));

    // 每个block的共享内存大小
    int sharedMemSize = threadsPerBlock * sizeof(double); 
    // 调用reduceKernel
    reduceKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_errors, d_result, size);

    // 从GPU复制最终结果回CPU
    double *h_result = new double[blocksPerGrid]; 
    cudaMemcpy(h_result, d_result, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    // 在CPU上完成最终的规约
    double error = 0.0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        error += h_result[i];
    }

    // Free device memory
    cudaFree(d_candidate);
    cudaFree(d_errors);
    cudaFree(d_result);

    // Continue with the rest of the test function
    if (error < bestsofar)
    {
        ++full_dtw;
        const double trueerror = mDTW.fastdynamic_origin(V, candidate);
        if (trueerror < bestsofar)
            bestsofar = trueerror;
    }
    return bestsofar;
}

double LB_Keogh::test(double* candidate)
{
    ++lb_keogh;
    double error(0.0);
    for (uint i = 0; i < size; ++i)
    {
        if (candidate[i] > U[i])
            error += candidate[i] - U[i];
        else if (candidate[i] < L[i])
            error += L[i] - candidate[i];
    }
    if (error < bestsofar)
    {
        ++full_dtw;
        const double trueerror =
            mDTW.fastdynamic(V, candidate);
        if (trueerror < bestsofar)
            bestsofar = trueerror;
    }
    return bestsofar;
}

double LB_Keogh::getLowestCost() { return bestsofar; }

LB_Keogh::LB_Keogh(double* v, unsigned int v_size, unsigned int constraint)
    : NearestNeighbor(v, constraint), size(v_size), lb_keogh(0), full_dtw(0)
{
    cudaMalloc(&V_K, size * sizeof(double));
    cudaMalloc(&U_K, size * sizeof(double));
    cudaMalloc(&L_K, size * sizeof(double));

    cudaMemcpy(V, V_K, size * sizeof(double), cudaMemcpyHostToDevice);
    
    assert(mConstraint >= 0);
    assert(mConstraint < static_cast<int>(size));

    Envelope envelope(V_K, U_K, L_K, size, mConstraint);
    envelope.compute();

    cudaMemcpy(U_K, U, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L_K, L, size * sizeof(double), cudaMemcpyDeviceToHost);
}

LB_Keogh::~LB_Keogh()
{
    cudaFree(V_K);
    cudaFree(U_K);
    cudaFree(L_K);
}