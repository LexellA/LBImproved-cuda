#include <cuda_runtime.h>

#include "NearestNeighbor.h"
#include "dtw.h"
#include "LB_Keogh.h" 
#include "Envelope.h"



double LB_Keogh::test(const double* candidate)
{
    ++lb_keogh;

    // 将数据从CPU复制到GPU
    cudaMemcpy(d_candidate, candidate, size * sizeof(double), cudaMemcpyHostToDevice);

    // 调用computeErrorKernel，计算出每个点的误差errors
    int threadsPerBlock = BLOCK_SZ;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;


    computeErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(
        U_K, L_K, d_candidate, d_errors, size);


    // 调用reduceKernel
    cudaMemset(d_result, 0, sizeof(double));
    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_errors, d_result, size);

    // 从GPU复制最终结果回CPU
    double error = 0;
    cudaMemcpy(&error, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout<<"error: "<<error<<std::endl;

    // Continue with the rest of the test function
    if (error < bestsofar)
    {
        ++full_dtw;
        const double trueerror = mDTW.fastdynamic(V_K, d_candidate);
        if (trueerror < bestsofar)
            bestsofar = trueerror;
    }

    return bestsofar;
}


double LB_Keogh::getLowestCost() { return bestsofar; }

LB_Keogh::LB_Keogh(double* v, unsigned int v_size, unsigned int constraint)
    : NearestNeighbor(v, v_size, constraint), size(v_size), lb_keogh(0), full_dtw(0), bestsofar(dtw::INF)
{
    cudaMalloc(&V_K, size * sizeof(double));
    cudaMalloc(&U_K, size * sizeof(double));
    cudaMalloc(&L_K, size * sizeof(double));
    cudaMalloc(&d_candidate, size * sizeof(double));
    cudaMalloc(&d_errors, size * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));

    cudaMemcpy(V_K, v, size * sizeof(double), cudaMemcpyHostToDevice);

    Envelope envelope(V_K, U_K, L_K, size, constraint);
    envelope.compute();
}

LB_Keogh::~LB_Keogh()
{
    cudaFree(V_K);
    cudaFree(U_K);
    cudaFree(L_K);

    cudaFree(d_candidate);
    cudaFree(d_errors);
    cudaFree(d_result);
}