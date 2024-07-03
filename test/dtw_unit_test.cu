#include <vector>
#include <chrono>
#include "../src/include/dtw_origin.h"
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
  Origin::dtw oDTW(x.size(), 1);
  mDTW.fastdynamic(d_x, d_y);
  oDTW.fastdynamic(x ,y);
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

  dtw mDTW(size ,size/10);
  Origin::dtw oDTW(size, size /10);
  double answer;

  auto start = std::chrono::high_resolution_clock::now();
  answer = mDTW.fastdynamic(d_v ,d_w);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "GPU time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                   .count()
            << "ms " 
            << "fd: "
            << answer << std::endl;

  start = std::chrono::high_resolution_clock::now();
  answer = mDTW.fastdynamic_SC(d_v ,d_w);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "GPU time with CUDA Graph: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                   .count()
            << "ms " 
            << "fd: "
            << answer << std::endl;

  start = std::chrono::high_resolution_clock::now();
  answer = oDTW.fastdynamic(v ,w);
  end = std::chrono::high_resolution_clock::now();
  std::cout << "CPU time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                   .count()
            << "ms " 
            << "fd: "
            << answer << std::endl;
}


int main(){
  //unit_test_1();
  unit_test_2(13000);

  return 0;
}
