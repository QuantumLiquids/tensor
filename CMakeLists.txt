# SPDX-License-Identifier: LGPL-3.0-only
#
# Author: Rongyang Sun <sun-rongyang@outlook.com>
# Creation Date: 2019-04-24 16:28
#
# Description: GraceQ/tensor project. Main CMake file.
#


# CMake basic setting
cmake_minimum_required(VERSION 3.12)

project("GraceQ/tensor"
  DESCRIPTION "A high-performance tensor computation framework for the quantum physics community"
  LANGUAGES CXX
)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

# Build type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(CMAKE_VERBOSE_MAKEFILE ON)
endif()


# Advanced options.
option(GQTEN_BUILD_UNITTEST "Build unittests for GraceQ/tensor." OFF)

option(GQTEN_TIMING_MODE "Timing mode." ON)
if(GQTEN_TIMING_MODE)
  add_definitions(-DGQTEN_TIMING_MODE)
endif()

option(GQTEN_MPI_TIMING_MODE "Timing mode." ON)
if(GQTEN_MPI_TIMING_MODE)
  add_definitions(-DGQTEN_MPI_TIMING_MODE)
endif()

option(GQTEN_USE_EXTERNAL_HPTT_LIB "Use external hptt library" OFF)


# Compilation and linking control.
set(CMAKE_CXX_STANDARD 17)
add_definitions(-Wall -g)


if(APPLE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
    #Need test
    set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl)
  endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
    # May not work
    set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -L$ENV{MKLROOT}/lib -L$ENV{CMPLR_ROOT}/mac/compiler/lib/ -liomp5 -lpthread -lm -ldl)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
    # Note as of Intel oneAPI 2021.2, "source /opt/intel/oneapi/compiler/latest/env/vars.sh"
    set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -L$ENV{MKLROOT}/lib -L$ENV{CMPLR_ROOT}/mac/compiler/lib/ -Wl, -rpath $ENV{CMPLR_ROOT}/mac/compiler/lib/libiomp5.dylib -liomp5 -lpthread -lm -ldl)
  endif()
elseif(UNIX)
  #UNIX include APPLE, but we except it here
  if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
    set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
    # Link the Intel's OpenMP library to avoid performance issue when the library calls the MKL's gesdd function.
    set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L$ENV{MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
    set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L$ENV{MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl)
  endif()
endif()



# Build dependencies.
add_subdirectory(external)


# Installation.
set(GQTEN_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
set(GQTEN_HEADER_DIR ${GQTEN_HEADER_PATH}/gqten)
set(GQTEN_INCLUDE_DEST "${CMAKE_INSTALL_PREFIX}/include")
install(
  DIRECTORY ${GQTEN_HEADER_DIR}
  DESTINATION ${GQTEN_INCLUDE_DEST}
)


# Unittests
if(GQTEN_BUILD_UNITTEST)
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(tests tests)
endif()
