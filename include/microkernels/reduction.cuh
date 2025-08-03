#pragma once
#include <cuda_runtime.h>

namespace microkernels {

template <typename T>
__global__ void block_reduce_sum(const T* __restrict__ input, T* __restrict__ output, int N) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}   

template <typename T>
T reduce_sum(const T* d_input, int N, cudaStream_t stream = 0) {
    const int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Allocate intermediate output buffer (one partial sum per block)
    T* d_intermediate;
    cudaMalloc(&d_intermediate, blocks * sizeof(T));

    // First stage: reduce input to partial sums
    size_t shared_size = threads * sizeof(T);
    block_reduce_sum<<<blocks, threads, shared_size, stream>>>(
        d_input, d_intermediate, N);

    // Second stage: reduce partials (recurse until down to 1 block)
    int curr_N = blocks;
    while (curr_N > 1) {
        int next_blocks = (curr_N + threads - 1) / threads;
        T* d_temp;
        cudaMalloc(&d_temp, next_blocks * sizeof(T));

        shared_size = threads * sizeof(T);
        block_reduce_sum<<<next_blocks, threads, shared_size, stream>>>(
            d_intermediate, d_temp, curr_N);

        cudaFree(d_intermediate);
        d_intermediate = d_temp;
        curr_N = next_blocks;
    }

    // Copy final result from device to host
    T result;
    cudaMemcpyAsync(&result, d_intermediate, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_intermediate);

    return result;
}

template <typename T>
T* reduce_sum_partial(const T* d_input, int N, int& num_partials, cudaStream_t stream = 0) {
    const int threads = 256;
    num_partials = (N + threads - 1) / threads;

    T* d_output;
    cudaMalloc(&d_output, num_partials * sizeof(T));
    size_t shared_size = threads * sizeof(T);

    block_reduce_sum<<<num_partials, threads, shared_size, stream>>>(d_input, d_output, N);
    return d_output; // caller is responsible for cudaFree
}


template <typename T>
T reduce_sum_final(const T* d_input, int N, cudaStream_t stream = 0) {
    const int threads = 256;
    int curr_N = N;

    const T* d_curr_input = d_input;
    T* d_intermediate = nullptr;

    while (curr_N > 1) {
        int blocks = (curr_N + threads - 1) / threads;
        size_t shared_size = threads * sizeof(T);

        T* d_output;
        cudaMalloc(&d_output, blocks * sizeof(T));
        block_reduce_sum<<<blocks, threads, shared_size, stream>>>(
            d_curr_input, d_output, curr_N);

        if (d_curr_input != d_input) cudaFree((void*)d_curr_input);
        d_curr_input = d_output;
        curr_N = blocks;
    }

    // Copy final result to host
    T result;
    cudaMemcpyAsync(&result, d_curr_input, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree((void*)d_curr_input);

    return result;
}

} // namespace microkernels
