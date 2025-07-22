#pragma once
#include <cuda_runtime.h>

namespace ttk {

// GPU kernel: C[i] = A[i] + B[i]
template<typename T>
__global__ void add_kernel(const T* A, const T* B, T* C, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) C[idx] = A[idx] + B[idx];
}

// Host wrapper
template<typename T>
inline void add(const T* A, const T* B, T* C, int N) {
  constexpr int BS = 256;
  int GS = (N + BS - 1) / BS;
  add_kernel<T><<<GS, BS>>>(A, B, C, N);
  cudaDeviceSynchronize();
}

}  // namespace ttk
