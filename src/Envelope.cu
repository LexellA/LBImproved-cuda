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

void Envelope::compute() {

  int blockSize = BLOCK_SZ;
  int numBlocks = mSize;

  computeEnvelopeKernel<<<numBlocks, blockSize>>>(d_array, mSize, mConstraint,
                                                  d_maxvalues, d_minvalues);
  return;
}

__global__ void computeEnvelopeKernel(const double *array, unsigned int size,
                               unsigned int constraint, double *maxvalues,
                               double *minvalues) {
  __shared__ double sdataMax[Envelope::BLOCK_SZ];
  __shared__ double sdataMin[Envelope::BLOCK_SZ];

  unsigned int bid = blockIdx.x;
  unsigned int tid = threadIdx.x;

  sdataMax[tid] = -DBL_MAX;
  sdataMin[tid] = DBL_MAX;

  if (bid < size) {
    unsigned int start = bid > constraint ? bid - constraint : 0;
    unsigned int end = min(size, bid + constraint + 1);

    double maxval = array[start];
    double minval = array[start];

    for (int i = start + tid; i < end; i += blockDim.x) {
      maxval = max(maxval, array[i]);
      minval = min(minval, array[i]);
    }

    sdataMax[tid] = maxval;
    sdataMin[tid] = minval;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        sdataMax[tid] = max(sdataMax[tid], sdataMax[tid + s]);
        sdataMin[tid] = min(sdataMin[tid], sdataMin[tid + s]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      maxvalues[bid] = sdataMax[0];
      minvalues[bid] = sdataMin[0];
    }
  }
}