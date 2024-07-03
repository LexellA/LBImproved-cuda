#include <cuda_runtime.h>
#include <cfloat>

#include "Envelope.h"

Envelope::Envelope(double *d_array, double *d_maxvalues, double *d_minvalues,
                   unsigned int size, unsigned int constraint)
    : d_array(d_array),
      d_maxvalues(d_maxvalues),
      d_minvalues(d_minvalues),
      mSize(size),
      mConstraint(constraint) {}


Envelope::~Envelope() {}

void Envelope::compute(cudaStream_t& stream) {
  int blockSize = BLOCK_SZ;
  int numBlocks = (mSize + blockSize - 1) / blockSize;

  // computeEnvelopeKernel<<<numBlocks, blockSize,
  //                         (Envelope::BLOCK_SZ + 2 * mConstraint) *
  //                         sizeof(double)>>>(
  //     d_array, mSize, mConstraint, d_maxvalues, d_minvalues);

  computeEnvelopeKernelUsingCache<<<numBlocks, blockSize, 0, stream>>>(
      d_array, mSize, mConstraint, d_maxvalues, d_minvalues);
  
  return;
}

void Envelope::compute() {
  int blockSize = BLOCK_SZ;
  int numBlocks = (mSize + blockSize - 1) / blockSize;

  computeEnvelopeKernel<<<numBlocks, blockSize,
                          (Envelope::BLOCK_SZ + 2 * mConstraint) *
                          sizeof(double)>>>(
      d_array, mSize, mConstraint, d_maxvalues, d_minvalues);

  return;
}

__global__ void computeEnvelopeKernel(const double *array, unsigned int size,
                                      unsigned int constraint,
                                      double *maxvalues, double *minvalues) {
  extern __shared__ double sharedData[];

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  int start = gid - constraint;
  int sharedSize = Envelope::BLOCK_SZ + 2 * constraint;

  for (int i = 0; i < sharedSize - tid; i += blockDim.x) {
    int index = start + i;
    if (index < 0) {
      sharedData[i + tid] = array[0];
    } else if (index >= size) {
      sharedData[i + tid] = array[size - 1];
    } else {
      sharedData[i + tid] = array[index];
    }
  }

  __syncthreads();

  if (gid < size) {
    double maxvalue = -DBL_MAX;
    double minvalue = DBL_MAX;
    for (int i = 0; i < 2 * constraint + 1; i++) {
      if (sharedData[tid + i] > maxvalue) {
        maxvalue = sharedData[tid + i];
      }
      if (sharedData[tid + i] < minvalue) {
        minvalue = sharedData[tid + i];
      }
    }
    maxvalues[gid] = maxvalue;
    minvalues[gid] = minvalue;
  }
}

__global__ void computeEnvelopeKernelUsingCache(const double *array, unsigned int size,
                                      unsigned int constraint,
                                      double *maxvalues, double *minvalues) {
  __shared__ double sharedData[Envelope::BLOCK_SZ];

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if(gid < size){
    sharedData[tid] = array[gid];
  } else {
    sharedData[tid] = array[size - 1];
  }

  __syncthreads();

  if (gid < size) {
    double maxvalue = -DBL_MAX;
    double minvalue = DBL_MAX;

    int start = gid > constraint ? gid - constraint : 0;
    int end = min(gid + constraint + 1, size);

    int sharedStart = bid * blockDim.x;
    int sharedEnd = sharedStart + blockDim.x;

    for (int i = start; i < end; i++) {
      if (sharedStart <= i && i < sharedEnd) {
        if (sharedData[i - sharedStart] > maxvalue) {
          maxvalue = sharedData[i - sharedStart];
        }
        if (sharedData[i - sharedStart] < minvalue) {
          minvalue = sharedData[i - sharedStart];
        }
      } else {
        if (array[i] > maxvalue) {
          maxvalue = array[i];
        }
        if (array[i] < minvalue) {
          minvalue = array[i];
        }
      }
    }

    maxvalues[gid] = maxvalue;
    minvalues[gid] = minvalue;
  }
}