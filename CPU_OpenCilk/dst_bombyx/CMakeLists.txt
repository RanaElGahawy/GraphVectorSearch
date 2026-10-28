cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_COMPILER "/opt/opencilk/bin/clang++")
# OpenCilk's clang 16 cannot parse the libc++ headers of the macOS 26 SDK
if(APPLE AND EXISTS "/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk")
  set(CMAKE_OSX_SYSROOT "/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk")
endif()

project(dst CXX)

set(CMAKE_BUILD_TYPE Release)

add_executable(dst main.cpp)

target_compile_options(dst PRIVATE -fopencilk -Wno-unknown-pragmas)
target_link_options(dst PRIVATE -fopencilk)
