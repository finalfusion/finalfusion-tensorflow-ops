cmake_minimum_required(VERSION 3.0)

project(finalfusion_tf)

enable_testing()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(COPY tf_flags.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
execute_process(COMMAND python ${CMAKE_CURRENT_BINARY_DIR}/tf_flags.py OUTPUT_VARIABLE TF_CFG ERROR_QUIET)
list(GET TF_CFG 2 CXX_11_ABI)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${CXX_11_ABI})

add_subdirectory(finalfusion-cxx)
add_subdirectory(finalfusion-tf)

