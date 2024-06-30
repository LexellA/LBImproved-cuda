#include <vector>
#include "../src/include/dtw.h"

std::vector<double> get_rand_seq(uint size) {
  std::vector<double> data(size);
  data[0] = 0.0;
  for (uint k = 1; k < size; ++k){
    data[k] = (1.0 * rand() / (RAND_MAX)) - 0.5 + data[k - 1];
    // std::cout << data[k] << " ";
  }
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
  dtw mDTW(x.size(), 1);
  mDTW.fastdynamic(x, y);
  mDTW.fastdynamic_origin(x ,y);
}

void unit_test_2(int size){
  std::vector<double> v ,w;
  v = get_rand_seq(size);
  w = get_rand_seq(size);
  dtw DTW(size ,1);
  DTW.fastdynamic(v ,w);
  DTW.fastdynamic_origin(v ,w);
}


int main(){
  //unit_test_1();
  unit_test_2(10000);

  return 0;
}
