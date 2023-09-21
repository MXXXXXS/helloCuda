#include "../../cudaByExample/common/book.h"

#define N (33 * 1024)

__global__ void add(int *a, int *b, int *c)
{
  // 这里都是一维的， 便于学习理解
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid == 0)
  {
    printf("3. Device calculating...\n");
  }
  while (tid < N)
  {
    c[tid] = a[tid] + b[tid];
    // 参考addVectors.cu里组合addP1, addP2的思想
    // 单次并行的数量： 单个block treads数量 x 所有block的数量
    tid += blockDim.x * gridDim.x;
  }
}

int main(void)
{
  int a[N], b[N], c[N];
  int *dev_a, *dev_b, *dev_c;

  HANDLE_ERROR(cudaMalloc((void **)&dev_a, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_b, N * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_c, N * sizeof(int)));

  for (int i = 0; i < N; i++)
  {
    a[i] = i;
    b[i] = i * i;
  }

  HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  printf("1. Call kernel\n");
  add<<<128, 128>>>(dev_a, dev_b, dev_c);

  printf("2. Called kernel\n");

  HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

  printf("4. Device calculate finished\n");

  // verify results
  bool success = true;
  for (int i = 0; i < N; i++)
  {
    if ((a[i] + b[i]) != c[i])
    {
      printf("Error: %d + %d != %d\n", a[i], b[i], c[i]);
      success = false;
      break;
    }
  }
  if (success)
    printf("Calculation successed\n");

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
  return 0;
}
