#include <iostream>
#include <chrono>
#include <functional>

#define N 10

__global__ void pAdd(int *a, int *b, int *c)
{
  int tid = blockIdx.x;
  if (tid < N)
  {
    c[tid] = a[tid] + b[tid];
  }
}

void addP1(int *a, int *b, int *c)
{
  int tid = 0;
  while (tid < N)
  {
    c[tid] = a[tid] + b[tid];
    tid += 2;
  }
}
void addP2(int *a, int *b, int *c)
{
  int tid = 1;
  while (tid < N)
  {
    c[tid] = a[tid] + b[tid];
    tid += 2;
  }
}

void add(int *a, int *b, int *c)
{
  int tid = 0;
  while (tid < N)
  {
    c[tid] = a[tid] + b[tid];
    tid += 1;
  }
}

void runPAdd(int *a, int *b, int *c)
{
  int *dev_a, *dev_b, *dev_c;
  cudaMalloc((void **)&dev_a, N * sizeof(int));
  cudaMalloc((void **)&dev_b, N * sizeof(int));
  cudaMalloc((void **)&dev_c, N * sizeof(int));

  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  pAdd<<<N, 1>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}

template <typename Func, typename... Args>
void timedFunction(Func func, Args &&...args)
{
  std::chrono::high_resolution_clock::time_point t1, t2;

  t1 = std::chrono::high_resolution_clock::now();

  func(std::forward<Args>(args)...);

  t2 = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);

  std::cout << "add time：" << duration.count() << "\n";
}

int main(void)
{

  int a[N], b[N], c[N];

  for (int i = 0; i < N; i++)
  {
    a[i] = -i;
    b[i] = i * i;
  }

  timedFunction(add, a, b, c);

  timedFunction(addP1, a, b, c);
  timedFunction(addP2, a, b, c);

  timedFunction(runPAdd, a, b, c);

  for (int i = 0; i < N; i++)
  {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  return 0;
}