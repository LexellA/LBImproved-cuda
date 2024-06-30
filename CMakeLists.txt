cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(LBImproved-cuda LANGUAGES CXX CUDA)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(LBImproved-cuda ${SOURCES})

target_include_directories(LBImproved-cuda PRIVATE ${PROJECT_SOURCE_DIR}/src/include)