cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(LBImproved-cuda LANGUAGES CXX CUDA)

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