
class Envelope {
 public:
  enum { BLOCK_SZ = 384 };
  Envelope(double* d_array, double* d_maxvalues, double* d_minvalues,
           unsigned int size, unsigned int constraint);
  ~Envelope();
  void compute(cudaStream_t& stream);

 private:
  double *d_array;
  double *d_maxvalues;
  double *d_minvalues;
  unsigned int mSize;
  unsigned int mConstraint;
};

__global__ void computeEnvelopeKernel(const double *array, unsigned int size,
                                      unsigned int constraint,
                                      double *maxvalues, double *minvalues);

__global__ void computeEnvelopeKernelUsingCache(const double *array,
                                                unsigned int size,
                                                unsigned int constraint,
                                                double *maxvalues,
                                                double *minvalues);