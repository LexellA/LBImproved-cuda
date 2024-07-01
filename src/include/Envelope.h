
class Envelope {
 public:
  enum { BLOCK_SZ = 256 };
  Envelope(double* d_array, double* d_maxvalues, double* d_minvalues,
           unsigned int size, unsigned int constraint);
  ~Envelope();
  void compute();

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