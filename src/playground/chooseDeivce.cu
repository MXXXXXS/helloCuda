#include <iostream>

int main(void)
{
  cudaDeviceProp prop;
  int dev;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 7;
  prop.minor = 5;
  cudaChooseDevice(&dev, &prop);
  printf("ID of CUDA device closest to revision %d.%d: %d\n", prop.major, prop.minor, dev);
}