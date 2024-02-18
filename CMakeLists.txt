cmake_minimum_required(VERSION 3.27)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)
set(CFLAGS -D__HIP_PLATFORM_AMD__)
set(CXXFLAGS D__HIP_PLATFORM_AMD__)

add_executable(NeuralNetwork
        main.cpp
)

include_directories("/opt/rocm/include")
