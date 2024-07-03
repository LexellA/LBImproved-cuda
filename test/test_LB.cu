#include <iostream>
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include "LB_Keogh.h"
#include "dtw_origin.h"

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

// Function to demo the LB_Keogh class with random walk data
void demo(uint size)
{
    std::cout << "Generating random walk and matching it with other random walks..." << std::endl;

    double *target = new double[size];
    getrandomwalk(target, size);
    LB_Keogh filter(target, size,  size / 10);               // Use DTW with a tolerance of 10% (size/10)
    double bestsofar = filter.getLowestCost();
    uint howmany = 10;

    std::vector<double> target_origin(target, target + size);
    Origin::LB_Keogh filter_origin(target_origin, size / 10);        // Use the original DTW with a tolerance of 10% (size/10
    
    // Allocate CUDA events for original test function
    cudaEvent_t start, stop;
    checkCudaStatus(cudaEventCreate(&start), "Failed to create start event");
    checkCudaStatus(cudaEventCreate(&stop), "Failed to create stop event");

    float totalMillisecondsTest = 0.0f;
    float totalMillisecondsTestKernel = 0.0f;

    for (uint i = 0; i < howmany; ++i)
    {
        double *candidate = new double[size];
        getrandomwalk(candidate, size);

        // Timing the test_kernel function
        checkCudaStatus(cudaEventRecord(start), "Failed to record start event");
        double newbestKernel = filter.test(candidate);
        checkCudaStatus(cudaEventRecord(stop), "Failed to record stop event");
        checkCudaStatus(cudaEventSynchronize(stop), "Failed to synchronize stop event");
        float millisecondsTestKernel = 0;
        checkCudaStatus(cudaEventElapsedTime(&millisecondsTestKernel, start, stop), "Failed to get elapsed time");
        totalMillisecondsTestKernel += millisecondsTestKernel;

        // Timing the original test function
        checkCudaStatus(cudaEventRecord(start), "Failed to record start event");
        std::vector<double> candidate_origin(candidate, candidate + size);
        double newbest = filter_origin.test(candidate_origin);
        checkCudaStatus(cudaEventRecord(stop), "Failed to record stop event");
        checkCudaStatus(cudaEventSynchronize(stop), "Failed to synchronize stop event");
        float millisecondsTest = 0;
        checkCudaStatus(cudaEventElapsedTime(&millisecondsTest, start, stop), "Failed to get elapsed time");
        totalMillisecondsTest += millisecondsTest;

        std::cout << i <<std::endl;
        assert(newbest == newbestKernel);

        if (newbest < bestsofar || newbestKernel < bestsofar)
        {
            // std::cout << "Found a new nearest neighbor, distance (L1 norm) = " << newbest << std::endl;
            bestsofar = newbest;
        }

        // std::cout << "Iteration: " << i + 1 << ", Time (test): " << millisecondsTest << " ms, Time (test_kernel): " << millisecondsTestKernel << " ms" << std::endl;
    }

    std::cout << "Compared with " << howmany << " random walks, closest match is at a distance (L1 norm) of " << filter.getLowestCost() << std::endl;
    std::cout << "Average time (test): " << totalMillisecondsTest / howmany << " ms" << std::endl;
    std::cout << "Average time (test_kernel): " << totalMillisecondsTestKernel / howmany << " ms" << std::endl;

    // Clean up
    checkCudaStatus(cudaEventDestroy(start), "Failed to destroy start event");
    checkCudaStatus(cudaEventDestroy(stop), "Failed to destroy stop event");
}

int main()
{
  demo(10);
  
    return 0;
}
