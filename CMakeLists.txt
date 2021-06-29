cmake_minimum_required(VERSION 3.17)
project(Homework_4 LANGUAGES CUDA CXX)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-ccbin \"C:/Program Files (x86)/Microsoft Visual Studio/2017/BuildTools/VC/Tools/MSVC/14.16.27023/bin/\" -arch \"sm_61\"")

add_executable(Homework_4 mp1_Nicholson-RubidouxCole.cu)

set_target_properties(
        Homework_4
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)