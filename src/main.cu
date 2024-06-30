#include <stdio.h>

__global__ void helloWorld() { printf("Hello, World from GPU!\n"); }

int main() {
  // Launch the kernel
  helloWorld<<<1, 10>>>();
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  return 0;
}