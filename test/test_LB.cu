#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "dtw.h"
#include "LB_Keogh.h"

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
std::vector<double> getrandomwalk(uint size)
{
    std::vector<double> data(size);
    data[0] = 0.0;
    for (uint k = 1; k < size; ++k)
        data[k] = (1.0 * rand() / RAND_MAX) - 0.5 + data[k - 1];
    return data;
}

// Function to demo the LB_Keogh class with random walk data
void demo(uint size)
{
    std::cout << "Generating random walk and matching it with other random walks..." << std::endl;

    std::vector<double> target = getrandomwalk(size); // This is our target
    LB_Keogh filter(target, size / 10);               // Use DTW with a tolerance of 10% (size/10)
    double bestsofar = filter.getLowestCost();
    uint howmany = 5000;

    // Allocate CUDA events for original test function
    cudaEvent_t start, stop;
    checkCudaStatus(cudaEventCreate(&start), "Failed to create start event");
    checkCudaStatus(cudaEventCreate(&stop), "Failed to create stop event");

    float totalMillisecondsTest = 0.0f;
    float totalMillisecondsTestKernel = 0.0f;

    for (uint i = 0; i < howmany; ++i)
    {
        std::vector<double> candidate = getrandomwalk(size);

        // Timing the original test function
        checkCudaStatus(cudaEventRecord(start), "Failed to record start event");
        double newbest = filter.test(candidate);
        checkCudaStatus(cudaEventRecord(stop), "Failed to record stop event");
        checkCudaStatus(cudaEventSynchronize(stop), "Failed to synchronize stop event");
        float millisecondsTest = 0;
        checkCudaStatus(cudaEventElapsedTime(&millisecondsTest, start, stop), "Failed to get elapsed time");
        totalMillisecondsTest += millisecondsTest;

        // Timing the test_kernel function
        checkCudaStatus(cudaEventRecord(start), "Failed to record start event");
        double newbestKernel = filter.test_kernel(candidate);
        checkCudaStatus(cudaEventRecord(stop), "Failed to record stop event");
        checkCudaStatus(cudaEventSynchronize(stop), "Failed to synchronize stop event");
        float millisecondsTestKernel = 0;
        checkCudaStatus(cudaEventElapsedTime(&millisecondsTestKernel, start, stop), "Failed to get elapsed time");
        totalMillisecondsTestKernel += millisecondsTestKernel;

        if (newbest < bestsofar || newbestKernel < bestsofar)
        {
            std::cout << "Found a new nearest neighbor, distance (L1 norm) = " << newbest << std::endl;
            bestsofar = newbest;
        }

        std::cout << "Iteration: " << i + 1 << ", Time (test): " << millisecondsTest << " ms, Time (test_kernel): " << millisecondsTestKernel << " ms" << std::endl;
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
    demo(10000);
    return 0;
}
