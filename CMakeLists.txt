cmake_minimum_required(VERSION 3.16)
project(scribblez CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json so editors (clangd, MS C/C++) can find headers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
  # py/build.py always passes this, set to the target CPU microarchitecture
  # (e.g. "alderlake", "x86-64-v3") so every translation unit is compiled for
  # that arch's instruction set instead of the generic x86-64 baseline.
  if(SCRIBBLEZ_MARCH)
    add_compile_options(-march=${SCRIBBLEZ_MARCH})
  endif()
endif()

# Link-time optimization for optimized builds (no-op where the toolchain lacks
# IPO support, e.g. some Debug setups).
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

# GoogleTest for the C++ test binaries (system libgtest-dev; the Docker image
# ships it prebuilt with its CMake config).
find_package(GTest REQUIRED)

enable_testing()
add_subdirectory(engine)
