# 3.30 is required for CXX_STANDARD 26 to resolve to -std=c++26 on GCC (the
# GNU compiler tables gate that mapping on CMake >= 3.30 and GCC >= 14); the
# Docker image's cmake is upgraded past Ubuntu 24.04's stock 3.28 to match.
cmake_minimum_required(VERSION 3.30)
project(scribblez CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# P2996 reflection, which GCC 16 gates behind a flag (data/format_layout.cpp
# derives the sidecar-format document from the structs themselves). Drop the
# flag once reflection stops being opt-in.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-freflection>)

# Compile through ccache, which the dev image installs and configures (see
# docker-setup/local/Dockerfile).
find_program(CCACHE_PROGRAM ccache REQUIRED)
set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})

# Emit compile_commands.json so editors (clangd, MS C/C++) can find headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Test: Release's optimization flags without link-time optimization, for
# build-and-test iteration. Tests exercise the same optimized code that ships,
# but every rebuild skips the LTO link, which dominates incremental Release
# builds. (Debug stays unoptimized, for debugging.)
set(CMAKE_CXX_FLAGS_TEST "${CMAKE_CXX_FLAGS_RELEASE}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)
  # py/build.py always passes this, set to the target CPU microarchitecture
  # (e.g. "alderlake", "x86-64-v3") so every translation unit is compiled for
  # that arch's instruction set instead of the generic x86-64 baseline.
  if(SCRIBBLEZ_MARCH)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=${SCRIBBLEZ_MARCH}>)
  endif()
endif()

# Link-time optimization for the shipped optimized builds; Test deliberately
# omits it (no-op where the toolchain lacks IPO support).
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
else()
  message(STATUS "LTO/IPO not supported: ${ipo_error}")
endif()

# Boost: program_options for the CLI, json for (de)serialization, and
# process/asio/filesystem (header-only, but they need filesystem+system+threads
# linked) for launching the web UI.
find_package(Boost REQUIRED COMPONENTS program_options json filesystem system)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Test binaries are only useful for an arch this machine can execute: one
# compiled for an arch whose instructions the host lacks (e.g. AVX-512 on a CPU
# without it) dies with SIGILL. py/build.py therefore builds tests for the
# host's own arch only; the other archs of a multi-arch build exist to be
# bundled to workers, and no test binary is ever bundled.
option(SCRIBBLEZ_BUILD_TESTS "Build the C++ test binaries" ON)

if(SCRIBBLEZ_BUILD_TESTS)
  # GoogleTest for the C++ test binaries (system libgtest-dev; the Docker image
  # ships it prebuilt with its CMake config).
  find_package(GTest REQUIRED)
  enable_testing()
endif()

add_subdirectory(engine)
