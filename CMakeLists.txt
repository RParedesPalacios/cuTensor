cmake_minimum_required(VERSION 3.18)
project(cuTensor LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(include)
include_directories(include/gpu)

file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES "src/gpu/*.cu")

add_library(cuTensor SHARED ${CPP_SOURCES} ${CUDA_SOURCES})
set_target_properties(cuTensor PROPERTIES OUTPUT_NAME "cuTensor")

set_target_properties(cuTensor PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuTensor PRIVATE CUDA::cublas pybind11::pybind11)

# If necessary, set specific CUDA architectures
# target_compile_options(cuTensor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_XX>)