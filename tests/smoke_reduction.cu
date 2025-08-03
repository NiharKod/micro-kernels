#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

#include "microkernels/reduction.cuh"

int main() {
    using T = float;
    const int N = 1024;

    // 1. Host input
    std::vector<T> h_input(N);
    std::iota(h_input.begin(), h_input.end(), 1); // [1, 2, ..., N]
    T cpu_result = std::accumulate(h_input.begin(), h_input.end(), T(0));

    // 2. Copy to device
    T* d_input;
    cudaMalloc(&d_input, N * sizeof(T));
    cudaMemcpy(d_input, h_input.data(), N * sizeof(T), cudaMemcpyHostToDevice);

    // 3. Partial reduction
    int num_partials;
    T* d_partials = microkernels::reduce_sum_partial(d_input, N, num_partials);

    std::vector<T> h_partials(num_partials);
    cudaMemcpy(h_partials.data(), d_partials, num_partials * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << "Partial sums per block:\n";
    for (int i = 0; i < num_partials; ++i) {
        std::cout << "  Block " << i << ": " << h_partials[i] << "\n";
    }

    // 4. Final reduction
    T gpu_result = microkernels::reduce_sum_final(d_input, N);
    std::cout << "\nGPU total sum: " << gpu_result << "\n";
    std::cout << "CPU total sum: " << cpu_result << "\n";

    // 5. Cleanup
    cudaFree(d_input);
    cudaFree(d_partials);

    return 0;
}
