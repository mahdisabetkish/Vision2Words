cmake_minimum_required(VERSION 3.20)
project(vision2words_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ONNX Runtime: vendored as a prebuilt package under third_party/ rather than
# fetched by CMake, since it ships prebuilt binaries (there's no source
# build happening here) -- set ONNXRUNTIME_ROOT to override if you've
# installed it elsewhere (e.g. a system package on Linux).
if(NOT DEFINED ONNXRUNTIME_ROOT)
    file(GLOB ORT_CANDIDATES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-*")
    list(GET ORT_CANDIDATES 0 ONNXRUNTIME_ROOT)
endif()
if(NOT ONNXRUNTIME_ROOT OR NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ONNX Runtime not found. Expected a prebuilt package under "
                         "third_party/onnxruntime-* (see cpp/README.md), or pass "
                         "-DONNXRUNTIME_ROOT=/path/to/onnxruntime.")
endif()
message(STATUS "Using ONNX Runtime at ${ONNXRUNTIME_ROOT}")

add_executable(v2w_caption
    src/main.cpp
    src/preprocess.cpp
    src/caption_model.cpp
)
target_include_directories(v2w_caption PRIVATE
    src
    third_party
    third_party/stb
    "${ONNXRUNTIME_ROOT}/include"
)
if(WIN32)
    target_link_libraries(v2w_caption PRIVATE "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
else()
    target_link_libraries(v2w_caption PRIVATE "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
endif()

# ONNX Runtime ships as a shared library; copy it next to the exe so
# v2w_caption runs without the caller having to fix up PATH/LD_LIBRARY_PATH.
file(GLOB ONNXRUNTIME_SHARED_LIBS
    "${ONNXRUNTIME_ROOT}/lib/*.dll"
    "${ONNXRUNTIME_ROOT}/lib/*.so*"
    "${ONNXRUNTIME_ROOT}/lib/*.dylib"
)
add_custom_command(TARGET v2w_caption POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ONNXRUNTIME_SHARED_LIBS} $<TARGET_FILE_DIR:v2w_caption>
)
