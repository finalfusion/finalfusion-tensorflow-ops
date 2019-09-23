cmake_minimum_required(VERSION 3.0)

project(finalfusion_tf)

enable_testing()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PythonInterp REQUIRED)

execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tf_flags.py
        OUTPUT_VARIABLE TF_CFG
        RESULT_VARIABLE TF_NOT_FOUND
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

if (TF_NOT_FOUND)
    message(FATAL_ERROR "Tensorflow not found")
endif ()

list(GET TF_CFG 2 CXX_11_ABI)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${CXX_11_ABI})

add_subdirectory(finalfusion-cxx)
add_subdirectory(finalfusion-tf)
add_subdirectory(tests)
