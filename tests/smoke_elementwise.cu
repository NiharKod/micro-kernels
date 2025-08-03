#include <iostream>
#include <cuda_runtime.h>
#include <microkernels/elementwise.cuh>

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << msg << ": " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 10;
    float h_in[N], h_out[N];

    // Fill input with values -5 to 4
    for (int i = 0; i < N; ++i) {
        h_in[i] = i - 5;
    }

    float *d_in = nullptr, *d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, N * sizeof(float)), "cudaMalloc d_in");
    check_cuda(cudaMalloc(&d_out, N * sizeof(float)), "cudaMalloc d_out");

    check_cuda(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Launch the ReLU kernel
    microkernels::launch_elementwise(d_out, d_in, N, microkernels::ReLU());

    // Ensure kernel launch was okay
    check_cuda(cudaGetLastError(), "kernel launch");

    check_cuda(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    check_cuda(cudaFree(d_in), "cudaFree d_in");
    check_cuda(cudaFree(d_out), "cudaFree d_out");

    std::cout << "Input:\n";
    for (int i = 0; i < N; ++i) std::cout << h_in[i] << " ";
    std::cout << "\n";

    std::cout << "ReLU Output:\n";
    for (int i = 0; i < N; ++i) std::cout << h_out[i] << " ";
    std::cout << "\n";

    return 0;
}
