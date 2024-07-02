#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <deque>
#include <chrono>

#include "NearestNeighbor.h"
#include "dtw.h"
#include "LB_Improved.h"
#include "LB_Keogh.h"
#include "Envelope.h"

static void computeEnvelope(const std::vector<double> &array, uint constraint, std::vector<double> &maxvalues, std::vector<double> &minvalues);

__global__ void computeErrorKernelBuffer(const double *U, const double *L, const double *candidate, double *errors, double *buffer, unsigned int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double temp = candidate[i];
    double upper = U[i];
    double lower = L[i];
    double error = 0, r_buffer = 0;
    if (i < size)
    {
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
        }
    }
    errors[i] = error;
    buffer[i] = r_buffer;
}



/**
 * 规约计算数组和
 */
uint compute_sum(double *d_array, uint size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(double);

    double *d_result;
    cudaMalloc(&d_result, blocksPerGrid * sizeof(double));

    reduceKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_array, d_result, size);
    // 从GPU复制最终结果回CPU
    double *h_result = new double[blocksPerGrid];
    cudaMemcpy(h_result, d_result, blocksPerGrid * sizeof(double), cudaMemcpyDeviceToHost);

    // 在CPU上完成最终的规约
    double sum = 0.0;
    for (int i = 0; i < blocksPerGrid; i++)
    {
        sum += h_result[i];
    }

    // Free device memory
    cudaFree(d_result);
    free(h_result);

    return sum;
}

double LB_Improved::test_kernel(const double *candidate)
{
    ++lb_keogh;

    // 分配device内存
    double *d_candidate, *d_errors, *d_buffer;
    // cudaMalloc(&d_V, size * sizeof(double));
    cudaMalloc(&d_candidate, size * sizeof(double));
    cudaMalloc(&d_errors, size * sizeof(double));
    cudaMalloc(&d_buffer, size * sizeof(double));

    // 将数据从CPU复制到GPU
    cudaMemcpy(d_candidate, candidate, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_buffer, candidate, size * sizeof(double), cudaMemcpyHostToDevice);

    // 第二次LB_Keogh
    double *d_U2, *d_L2;
    cudaMalloc(&d_U2, size * sizeof(double));
    cudaMalloc(&d_L2, size * sizeof(double));

    // 调用computeErrorKernel，计算出每个点的误差errors
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    // auto start = std::chrono::high_resolution_clock::now();

    computeErrorKernelBuffer<<<blocksPerGrid, threadsPerBlock>>>(U_K, L_K, d_candidate, d_errors, d_buffer, size);

    // cudaDeviceSynchronize();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // std::cout << "Time (computeErrorKernel): " << duration << " us" << std::endl;



    double error = compute_sum(d_errors, size);

    // Continue with the rest of the test function
    if (error < bestsofar)
    {
        Envelope envelope(d_buffer, d_U2, d_L2, size, mConstraint);
        envelope.compute();

        computeErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(d_U2, d_L2, V_K, d_errors, size);

        double error2 = compute_sum(d_errors, size);

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // std::cout << "Time (test_kernel): " << milliseconds << " ms" << std::endl;

        if (error2 < bestsofar)
        {
            ++full_dtw;
            const double trueerror = mDTW.fastdynamic(V_K, d_candidate);
            if (trueerror < bestsofar)
                bestsofar = trueerror;
        }
    }

    // Free device memory
    cudaFree(d_U2);
    cudaFree(d_L2);

    cudaFree(d_candidate);
    cudaFree(d_errors);
    cudaFree(d_buffer);

    return bestsofar;
}

double LB_Improved::test(const double *candidate)
{
        // double *d_U2, *d_L2,  *U2, *L2;
        // double * d_buffer;
        // U2 = new double[size];
        // L2 = new double[size];
        // cudaMalloc(&d_U2, size * sizeof(double));
        // cudaMalloc(&d_L2, size * sizeof(double));
        // cudaMalloc(&d_buffer, size * sizeof(double));

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);

    ++lb_keogh;
    double error(0.0);
    // double *buffer = new double[size];
    std::vector<double> buffer(size);
    for (uint i = 0; i < size; ++i)
    {
        const double &cdi(candidate[i]);
        if (cdi > U[i])
        {
            error += cdi - (buffer[i] = U[i]);
        }
        else if (cdi < L[i])
        {
            error += (buffer[i] = L[i]) - cdi;
        }
        else
            buffer[i] = cdi;
        // if (error > bestsofar)
        //     return bestsofar;
    }

    // if (error < bestsofar)
    // {

        // cudaMemcpy(d_buffer, buffer, size * sizeof(double), cudaMemcpyHostToDevice);

        // Envelope envelope(d_buffer, d_U2, d_L2, size, mConstraint);

        // envelope.compute();

        // cudaMemcpy(U2, d_U2, size * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(L2, d_L2, size * sizeof(double), cudaMemcpyDeviceToHost);
        std::vector<double> U2(size), L2(size);
        computeEnvelope(buffer, mConstraint, U2, L2);

        for (uint i = 0; i < size; ++i)
        {
            if (V[i] > U2[i])
            {
                error += V[i] - U2[i];
            }
            else if (V[i] < L2[i])
            {
                error += L2[i] - V[i];
            }
            // if (error > bestsofar)
            //     return bestsofar;
        }

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // std::cout << "Time (test): " << milliseconds << " ms" << std::endl;

        // double *d_candidate;
        // cudaMalloc(&d_candidate, size * sizeof(double));
        // cudaMemcpy(d_candidate, candidate, size * sizeof(double), cudaMemcpyHostToDevice);

        // if (error < bestsofar)
        // {
            ++full_dtw;
            // const double trueerror = mDTW.fastdynamic(V_K, d_candidate); 
            std::vector<double> w(candidate, candidate + size);
            const double trueerror = mDTW.fastdynamic_origin(V_original, w);
            if (trueerror < bestsofar)
                bestsofar = trueerror;
    //     }

        
    // }
    //     // cudaFree(d_U2);
        // cudaFree(d_L2);
        // cudaFree(d_buffer);
        // free(buffer);
        // free(U2);
        // free(L2);
    return bestsofar;
}

double LB_Improved::getLowestCost() { return bestsofar; }

LB_Improved::LB_Improved(double *v, unsigned int v_size, unsigned int constraint)
    : NearestNeighbor(v, v_size, constraint), size(v_size), lb_keogh(0), full_dtw(0), bestsofar(dtw::INF)
{
    V = v;
    U = new double[size];
    L = new double[size];
    cudaMalloc(&V_K, size * sizeof(double));
    cudaMalloc(&U_K, size * sizeof(double));
    cudaMalloc(&L_K, size * sizeof(double));

    cudaMemcpy(V_K, V, size * sizeof(double), cudaMemcpyHostToDevice);

    Envelope envelope(V_K, U_K, L_K, size, mConstraint);
    envelope.compute();

    cudaMemcpy(U, U_K, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(L, L_K, size * sizeof(double), cudaMemcpyDeviceToHost);

    V_original = std::vector<double>(v, v + size);   
    // U_original = std::vector<double>(U, U + size);
    // L_original = std::vector<double>(L, L + size);
}

LB_Improved::~LB_Improved()
{
    cudaFree(V_K);
    cudaFree(U_K);
    cudaFree(L_K);
    free(U);
    free(L);
}

static void computeEnvelope(const std::vector<double> &array, uint constraint, std::vector<double> &maxvalues, std::vector<double> &minvalues)
{
    uint width = 1 + 2 * constraint;
    std::deque<int> maxfifo, minfifo;
    maxfifo.push_back(0);
    minfifo.push_back(0);
    for (uint i = 1; i < array.size(); ++i)
    {
        if (i >= constraint + 1)
        {
            maxvalues[i - constraint - 1] = array[maxfifo.front()];
            minvalues[i - constraint - 1] = array[minfifo.front()];
        }
        if (array[i] > array[i - 1])
        { // overshoot
            maxfifo.pop_back();
            while (maxfifo.size() > 0)
            {
                if (array[i] <= array[maxfifo.back()])
                    break;
                maxfifo.pop_back();
            }
        }
        else
        {
            minfifo.pop_back();
            while (minfifo.size() > 0)
            {
                if (array[i] >= array[minfifo.back()])
                    break;
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
    for (uint i = array.size(); i <= array.size() + constraint; ++i)
    {
        if (i >= constraint + 1)
        {
            maxvalues[i - constraint - 1] = array[maxfifo.front()];
            minvalues[i - constraint - 1] = array[minfifo.front()];
        }
        if (i - maxfifo.front() >= width)
            maxfifo.pop_front();
        if (i - minfifo.front() >= width)
            minfifo.pop_front();
    }
}