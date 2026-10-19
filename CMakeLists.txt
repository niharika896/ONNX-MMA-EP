cmake_minimum_required(VERSION 3.14)
project(ONNXEP CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma")

if(DEFINED ENV{ONNXRUNTIME_DIR})
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
else()
    message(FATAL_ERROR "set ONNX Runtine env variable")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${ONNXRUNTIME_DIR}/include
)

link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(inference_app src/main.cc src/mma_ep.cc)
target_link_libraries(inference_app PRIVATE onnxruntime)