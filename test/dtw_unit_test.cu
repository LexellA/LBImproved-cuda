#include <vector>
#include "../src/include/dtw.h"

std::vector<double> get_rand_seq(uint size) {
  std::vector<double> data(size);
  data[0] = 0.0;
  for (uint k = 1; k < size; ++k){
    data[k] = (1.0 * rand() / (RAND_MAX)) - 0.5 + data[k - 1];
    //std::cout << data[k] << " ";
  }
  //std::cout << std::endl;
  return data;
}

void unit_test_1(){
  std::vector<double> x;
  x.push_back(1);
  x.push_back(2);
  x.push_back(3);
  x.push_back(1.1);
  x.push_back(1);
  x.push_back(9);
  x.push_back(1.01);
  x.push_back(11);
  x.push_back(0.001);
  std::vector<double> y;
  for (uint k = 0; k < x.size(); ++k)
    y.push_back(x[k] + 15);
  
  double *d_x, *d_y;
  checkCudaErrors(cudaMalloc(&d_x, x.size() * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y, y.size() * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice));

  dtw mDTW(x.size(), 1);
  mDTW.fastdynamic(d_x, d_y);
  mDTW.fastdynamic_origin(x ,y);
}

void unit_test_2(int size){
  std::vector<double> v ,w;
  v = get_rand_seq(size);
  w = get_rand_seq(size);


  double *d_v, *d_w;
  checkCudaErrors(cudaMalloc(&d_v, size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_w, size * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_v, v.data(), size * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_w, w.data(), size * sizeof(double), cudaMemcpyHostToDevice));

  dtw DTW(size ,size/10);
  DTW.fastdynamic(d_v ,d_w);
  // DTW.fastdynamic_SC(d_v ,d_w);
  DTW.fastdynamic_origin(v ,w);
}


int main(){
  //unit_test_1();
  unit_test_2(13000);

  return 0;
}
