#pragma once
#include <cuda_runtime.h>

namespace microkernels {

template <typename T, typename F>
__global__ void elementwise_kernel(T* out, const T* in, int N, F op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = op(in[i]);
}

template <typename T, typename F>
void launch_elementwise(T* out, const T* in, int N, F op, cudaStream_t stream = 0) {
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    elementwise_kernel<<<gridSize, blockSize, 0, stream>>>(out, in, N, op);
}

struct ReLU {
    __device__ float operator()(float x) const { return x > 0 ? x : 0; }
};

}
