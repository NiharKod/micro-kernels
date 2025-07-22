cmake_minimum_required(VERSION 3.18)
project(TinyTensorKernels LANGUAGES CUDA CXX)

#  -- HEADER-ONLY interface library --
add_library(tiny_tensor_kernels INTERFACE)
target_include_directories(tiny_tensor_kernels INTERFACE
  ${PROJECT_SOURCE_DIR}/include
)

#  -- optional test build --
enable_testing()
add_executable(smoke_test tests/smoke_test.cu)
target_link_libraries(smoke_test PRIVATE tiny_tensor_kernels)
add_test(NAME smoke_test COMMAND smoke_test)

