project(libonnx)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_FLAGS "-std=gnu99 -pg")

file(GLOB sources src/*.c src/default/*.c)
message(STATUS "sources = ${sources}")
include_directories(src)
add_library(onnx STATIC ${sources})

add_executable(main main.c)
target_link_libraries(main onnx jpeg m)


# add_executable(test_slice tests/test_slice.c)
# target_link_libraries(test_slice onnx jpeg m)

add_executable(test_split tests/test_split.c)
target_link_libraries(test_split onnx jpeg m)
