cmake_minimum_required(VERSION 3.14)
project(llvc_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the ONNX Runtime setup script
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupOnnxRuntime.cmake)

# TinyWav
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/../tinywav)

# Main executable
add_executable(llvc_test 
    src/main.cpp
    ../lib/tinywav/tinywav.c
    ../lib/tinywav/myk_tiny.cpp
)

target_include_directories(llvc_test PRIVATE ${BACKEND_BUILD_HEADER_DIRS})
target_link_directories(llvc_test PRIVATE ${BACKEND_BUILD_LIBRARY_DIRS})
target_link_libraries(llvc_test onnxruntime)

# Copy ONNX Runtime shared library to build directory
if(WIN32)
    add_custom_command(TARGET llvc_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:llvc_test>)
endif()