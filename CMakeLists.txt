cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(LBImproved-cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(LBImproved-cuda ${SOURCES})
target_include_directories(LBImproved-cuda PRIVATE ${PROJECT_SOURCE_DIR}/src/include)

target_compile_options(LBImproved-cuda PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)





enable_testing()

add_executable(test_envelope test/test_envelope.cu src/Envelope.cu)
target_include_directories(test_envelope PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
target_compile_options(test_envelope PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
add_test(NAME test_envelope COMMAND test_envelope)


add_executable(test_dtw  src/include/dtw.h
                    src/dtw.cu
                    test/dtw_unit_test.cu)
target_include_directories(test_dtw PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
target_compile_options(test_dtw PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
add_test(NAME test_dtw COMMAND test_dtw)

add_executable(test_LB src/include/dtw.h
                        src/dtw.cu
                        src/LB_Keogh.cu
                        src/Envelope.cu
                        src/NearestNeighbor.cu
                        test/test_LB.cu)
target_include_directories(test_LB PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
target_compile_options(test_LB PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
add_test(NAME test_LB COMMAND test_LB)


add_executable(test_LBImproved 
                        src/dtw.cu
                        src/LB_Improved.cu
                        src/LB_Keogh.cu
                        src/Envelope.cu
                        test/test_LBImproved.cu)
target_include_directories(test_LBImproved PRIVATE ${PROJECT_SOURCE_DIR}/src/include)
target_compile_options(test_LBImproved PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
add_test(NAME test_LBImproved COMMAND test_LBImproved)