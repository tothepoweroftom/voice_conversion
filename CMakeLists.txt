cmake_minimum_required(VERSION 3.14)
project(llvc_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include the ONNX Runtime setup script
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupOnnxRuntime.cmake)

# TinyWav
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/../tinywav)

# Option to use portaudio
option(USE_PORTAUDIO "Use PortAudio" ON)




if(USE_PORTAUDIO)
    add_executable(llvc_test_pa 
    src/main_pa_threading.cpp
    ../lib/tinywav/tinywav.c
    ../lib/tinywav/myk_tiny.cpp
    )
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
    include_directories(${PORTAUDIO_INCLUDE_DIRS})
    target_include_directories(llvc_test_pa PRIVATE ${BACKEND_BUILD_HEADER_DIRS} ${PORTAUDIO_INCLUDE_DIRS})
    target_link_directories(llvc_test_pa PRIVATE ${BACKEND_BUILD_LIBRARY_DIRS} ${PORTAUDIO_LIBRARY_DIRS})
    target_link_libraries(llvc_test_pa onnxruntime ${PORTAUDIO_LIBRARIES})
    message(STATUS "Using PortAudio")
    # Main executable

else()
    add_executable(llvc_test 
    src/main.cpp
    ../lib/tinywav/tinywav.c
    ../lib/tinywav/myk_tiny.cpp
    )
    target_include_directories(llvc_test PRIVATE ${BACKEND_BUILD_HEADER_DIRS})
    target_link_directories(llvc_test PRIVATE ${BACKEND_BUILD_LIBRARY_DIRS})
    target_link_libraries(llvc_test onnxruntime)
    # Main executable

endif()




# Copy ONNX Runtime shared library to build directory
if(WIN32)
    add_custom_command(TARGET llvc_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ANIRA_ONNXRUNTIME_SHARED_LIB_PATH}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:llvc_test>)
endif()