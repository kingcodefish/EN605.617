cmake_minimum_required(VERSION 3.24)
project(Assignment2 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Assignment2 main.cu)

set_target_properties(
        Assignment2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native)