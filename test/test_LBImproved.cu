#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <chrono>
#include "LB_Improved.h"

// Helper function to check CUDA error status
void checkCudaStatus(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to generate random walk data
void getrandomwalk(double *data, uint size)
{
    data[0] = 0.0;
    for (uint k = 1; k < size; ++k)
        data[k] = (1.0 * rand() / RAND_MAX) - 0.5 + data[k - 1];
}

__global__ void warmupKernel()
{
    // to warmup
}

void warmupGPU()
{
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize(); // 确保核函数执行完成
}

int test(int size)
{
    std::cout << "Generating random walk and matching it with other random walks..." << std::endl;

    double *target = new double[size];
    getrandomwalk(target, size);
    LB_Improved filter_kernel(target, size, size / 10); // Use DTW with a tolerance of 10% (size/10)
    LB_Improved filter(target, size, size / 10); // Use DTW with a tolerance of 10% (size/10)
    // double bestsofar = filter.getLowestCost();
    // uint howmany = 1;

    // Allocate CUDA events for original test function



    double *candidate = new double[size];
    getrandomwalk(candidate, size);

    warmupGPU();

    // Timing the test_kernel function
    auto startKernel = std::chrono::high_resolution_clock::now();
    double bestKernel = filter_kernel.test_kernel(candidate);
    cudaDeviceSynchronize();
    auto endKernel = std::chrono::high_resolution_clock::now();
    auto durationKernel = std::chrono::duration_cast<std::chrono::milliseconds>(endKernel - startKernel).count();


    // Timing the original test function
    auto startTest = std::chrono::high_resolution_clock::now();
    double best = filter.test(candidate);
    auto endTest = std::chrono::high_resolution_clock::now();
    auto durationTest = std::chrono::duration_cast<std::chrono::milliseconds>(endTest - startTest).count();

    assert(best == bestKernel);


    // std::cout << "Iteration: " << i + 1 << ", Time (test): " << millisecondsTest << " ms, Time (test_kernel): " << millisecondsTestKernel << " ms" << std::endl;

    std::cout << "Average time (test): " << durationTest << " ms" << std::endl;
    std::cout << "Average time (test_kernel): " << durationKernel << " ms" << std::endl;

    // Clean up
    // checkCudaStatus(cudaEventDestroy(start), "Failed to destroy start event");
    // checkCudaStatus(cudaEventDestroy(stop), "Failed to destroy stop event");

    return 0;
}

int main()
{
    return test(10000);
}