cmake_minimum_required(VERSION 3.12)
project(microkernels LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Include the header-only library
include_directories(include)

# Add smoke test executable
add_executable(smoke_elementwise tests/smoke_elementwise.cu)
add_executable(smoke_reduction tests/smoke_reduction.cu)