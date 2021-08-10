#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Rongyang Sun <sun-rongyang@outlook.com>
#  Creation Date: 2019-04-24 21:16
#  
#  Description: GraceQ/tensor project. CMake file to control unittest.
# 
if(NOT GQTEN_USE_EXTERNAL_HPTT_LIB)
  set(hptt_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/external/hptt/include")
  set(hptt_LIBRARY "${CMAKE_BINARY_DIR}/external/hptt/libhptt.a")
endif()

find_package(Boost
  1.76.0
  REQUIRED 
  COMPONENTS serialization mpi)

find_package(MPI REQUIRED)

option(TEST_ACTUAL_COMBAT "test the actual examples of mps tensors" OFF)

if(TEST_ACTUAL_COMBAT)
  add_definitions(-DACTUALCOMBAT)
endif()

#set omp flag
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(OMP_FLAGS -qopenmp)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(OMP_FLAGS -fopenmp)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(OMP_FLAGS -fopenmp)
endif()

macro(add_unittest
    TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS)
  add_executable(${TEST_NAME}
    ${TEST_SRC})

    target_compile_options(${TEST_NAME}
      PRIVATE ${CFLAGS}
      PRIVATE ${OMP_FLAGS}
    )
    target_include_directories(${TEST_NAME}
      PRIVATE ${GQTEN_HEADER_PATH}
      PRIVATE ${hptt_INCLUDE_DIR}
      PRIVATE ${Boost_INCLUDE_DIR}
      PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
      GTest::GTest GTest::Main
      ${hptt_LIBRARY}
      Boost::serialization
      Boost::mpi
      ${MPI_CXX_LINK_FLAGS}
      ${MPI_mpi_LIBRARY}
      "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
      )

    add_test(
      NAME ${TEST_NAME}
      COMMAND "${TEST_NAME}")

  set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()



macro(add_mpi_unittest
    TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS PROCESSOR_NUM)
  add_executable(${TEST_NAME}
    ${TEST_SRC})

    target_compile_options(${TEST_NAME}
      PRIVATE ${CFLAGS} 
      PRIVATE ${OMP_FLAGS}
    )
    target_include_directories(${TEST_NAME}
      PRIVATE ${GQTEN_HEADER_PATH}
      PRIVATE ${hptt_INCLUDE_DIR}
      PRIVATE ${Boost_INCLUDE_DIR}
      PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
      GTest::GTest GTest::Main
      ${hptt_LIBRARY}
      Boost::serialization
      Boost::mpi
      ${MPI_CXX_LINK_FLAGS}
      ${MPI_mpi_LIBRARY}
      "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
      )

    if( NOT(MPIEXEC_MAX_NUMPROCS MATCHES "1"))
      add_test(
        NAME ${TEST_NAME}
        COMMAND ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} ${PROCESSOR_NUM} 
            $<TARGET_FILE:${TEST_NAME}>)
    endif()
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()


## Test quantum number related objects
# Test quantum number vaules
add_unittest(test_qnval_u1 "test_gqtensor/test_qnval/test_qnval_u1.cc" "" "" "")
# Test quantum number
add_unittest(test_qn "test_gqtensor/test_qn.cc" "" "" "")

## Test quantum number sector object.
add_unittest(test_qnsct "test_gqtensor/test_qnsct.cc" "" "" "")

## Test index object.
add_unittest(test_index "test_gqtensor/test_index.cc" "" "" "")

## Test symmetry-blocked sparse tensor, GQTensor.
add_unittest(test_gqtensor
  "test_gqtensor/test_gqtensor.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)

## Test tensor manipulation functions.
# Test basic tensor operations.
add_unittest(test_basic_operations
  "test_tensor_manipulation/test_basic_operations.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor linear combination.
add_unittest(test_ten_linear_combine
  "test_tensor_manipulation/test_ten_linear_combine.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor contraction.
add_unittest(test_ten_ctrct
  "test_tensor_manipulation/test_ten_ctrct.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor SVD.
add_unittest(test_ten_svd
  "test_tensor_manipulation/test_ten_svd.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor QR.
add_unittest(test_ten_qr
  "test_tensor_manipulation/test_ten_qr.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor index combination.
add_unittest(test_index_combine
  "test_tensor_manipulation/test_index_combine.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)
# Test tensor expansion
add_unittest(test_ten_expand
  "test_tensor_manipulation/test_ten_expand.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)

# Test tensor fuse index
add_unittest(test_ten_fuse_index
  "test_tensor_manipulation/test_ten_fuse_index.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)

# Test tensor comminutation for serialization
add_unittest(test_boost_serialization
  "test_gqtensor/test_boost_serialization.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)


# Test tensor comminutation for mpi
add_mpi_unittest(test_boost_mpi
  "test_gqtensor/test_boost_mpi.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "2"
)


#Test tensor restriction contract 
add_unittest(test_ten_ctrct_1sector
  "test_tensor_manipulation/test_ten_ctrct_1sector.cc"
  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
)