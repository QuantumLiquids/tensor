// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-28 19:07
*
* Description: GraceQ/tensor project. High performance BLAS Level 3 related
* functions based on MKL.
*/

/**
@file blas_level3.h
@brief High performance BLAS Level 3 related functions based on MKL.
*/
#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H


#include "gqten/framework/value_t.h"      // GQTEN_Double, GQTEN_Complex

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert

#include "mkl.h"      //cblas_*gemm


namespace gqten {

/// High performance numerical functions.
namespace hp_numeric {


inline void MatMultiply(
    const GQTEN_Double *a,
    const GQTEN_Double *b,
    const size_t m,
    const size_t k,
    const size_t n,
    const GQTEN_Double beta,
    GQTEN_Double *c) {
  cblas_dgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      1.0,
      a, k,
      b, n,
      beta,
      c, n
  );
}


inline void MatMultiply(
    const GQTEN_Complex *a,
    const GQTEN_Complex *b,
    const size_t m,
    const size_t k,
    const size_t n,
    const GQTEN_Complex beta,
    GQTEN_Complex *c) {
  GQTEN_Complex alpha(1.0);
  cblas_zgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k,
      &alpha,
      a, k,
      b, n,
      &beta,
      c, n
  );
}




inline void MatMultiplyBatch(
    const GQTEN_Double **a_array, const GQTEN_Double **b_array, 
    const MKL_INT *m_array, const MKL_INT *k_array, const MKL_INT *n_array,
    const GQTEN_Double *beta_array,
    GQTEN_Double **c_array, 
    const MKL_INT group_count) {

    const CBLAS_LAYOUT Layout = CblasRowMajor;
    const MKL_INT* lda_array = k_array;
    const MKL_INT *ldb_array = n_array;
    const MKL_INT *ldc_array = n_array;

#ifdef GQTEN_USE_MKL_GEMM_BATCH
  // NOTE: DONOT use this part code now, except contracting one index
  // because when c_array has some same elements (c_array[i]==c_array[j] with i!=j),
  //different CPU&caches will load&read the data at the same time.
  CBLAS_TRANSPOSE* transa_array = (CBLAS_TRANSPOSE *) malloc(group_count* sizeof(CBLAS_TRANSPOSE));
  CBLAS_TRANSPOSE* transb_array = (CBLAS_TRANSPOSE *) malloc(group_count* sizeof(CBLAS_TRANSPOSE));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    transa_array[i] = CblasNoTrans;
  }
  for (MKL_INT i = 0 ; i < group_count ; i++){
    transb_array[i] = CblasNoTrans;
  }

  GQTEN_Double *alpha_array = (GQTEN_Double *) malloc(group_count* sizeof(GQTEN_Double));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    alpha_array[i] = 1.0;
  }

  MKL_INT* group_size = (MKL_INT *) malloc(group_count* sizeof(MKL_INT));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    group_size[i] = 1;
  }

  cblas_dgemm_batch (
      Layout,
      transa_array, transb_array,
      m_array, n_array, k_array,
      alpha_array,
      a_array, lda_array,
      b_array, ldb_array,
      beta_array,
      c_array, ldc_array,
      group_count,
      group_size);

  free(transa_array);
  free(transb_array);
  free(alpha_array);
  free(group_size);
#else // Use direct gemm loop.

  auto idx = 0;
  for (MKL_INT i = 0; i < group_count; ++i) {
    for (MKL_INT j = 0; j < 1; ++j) {
      cblas_dgemm(
          Layout,
          CblasNoTrans, CblasNoTrans,
          m_array[i], n_array[i], k_array[i],
          1.0,
          a_array[idx], lda_array[i],
          b_array[idx], ldb_array[i],
          beta_array[i],
          c_array[idx], ldc_array[i]);
      ++idx;
    }
  }

#endif
}


inline void MatMultiplyBatch(
    const GQTEN_Complex **a_array, 
    const GQTEN_Complex **b_array, 
    const MKL_INT *m_array,  const MKL_INT *k_array, const MKL_INT *n_array,
    const GQTEN_Complex *beta_array,
    GQTEN_Complex **c_array,
    const MKL_INT group_count) {
    
    const CBLAS_LAYOUT Layout = CblasRowMajor;

    const MKL_INT* lda_array = k_array;
    const MKL_INT *ldb_array = n_array;
    const MKL_INT *ldc_array = n_array;

#ifdef GQTEN_USE_MKL_GEMM_BATCH
  // NOTE: DONOT use this part code now, except contracting one index
  // because when c_array has some same elements (c_array[i]==c_array[j] with i!=j),
  //different CPU&caches will load&read the data at the same time.
  CBLAS_TRANSPOSE* transa_array = (CBLAS_TRANSPOSE *) malloc(group_count* sizeof(CBLAS_TRANSPOSE));
  CBLAS_TRANSPOSE* transb_array = (CBLAS_TRANSPOSE *) malloc(group_count* sizeof(CBLAS_TRANSPOSE));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    transa_array[i] = CblasNoTrans;
  }
  for (MKL_INT i = 0 ; i < group_count ; i++){
    transb_array[i] = CblasNoTrans;
  }

  GQTEN_Complex *alpha_array = (GQTEN_Complex *) malloc(group_count* sizeof(GQTEN_Complex));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    alpha_array[i] = GQTEN_Complex(1.0);
  }

  MKL_INT* group_size = (MKL_INT *) malloc(group_count* sizeof(MKL_INT));
  for (MKL_INT i = 0 ; i < group_count ; i++){
    group_size[i] = 1;
  }

  const void** a_array_void_pointer = (const void** ) malloc(group_count* sizeof(void*));
  for(size_t i=0;i < group_count; i++){
    a_array_void_pointer[i] = (const void*) a_array[i];
  }
  const void** b_array_void_pointer = (const void** ) malloc(group_count* sizeof(void*));
  for(size_t i=0;i < group_count; i++){
    b_array_void_pointer[i] = (const void*) b_array[i];
  }
  void** c_array_void_pointer = (void** ) malloc(group_count* sizeof(void*));
  for(size_t i=0;i < group_count; i++){
    c_array_void_pointer[i] = (void*) c_array[i];
  }

  cblas_zgemm_batch (
      Layout,
      transa_array, transb_array,
      m_array, n_array, k_array,
      alpha_array,
      a_array_void_pointer, lda_array,
      b_array_void_pointer, ldb_array,
      beta_array,
      c_array_void_pointer, ldc_array,
      group_count,
      group_size);

  free(a_array_void_pointer);
  free(b_array_void_pointer);
  free(c_array_void_pointer);

  free(transa_array);
  free(transb_array);
  free(alpha_array);
  free(group_size);
#else // Use direct gemm loop.
  GQTEN_Complex alpha = GQTEN_Complex(1.0);
  auto idx = 0;
  for (MKL_INT i = 0; i < group_count; ++i) {
    for (MKL_INT j = 0; j < 1; ++j) {
      cblas_zgemm(
          Layout,
          CblasNoTrans, CblasNoTrans,
          m_array[i], n_array[i], k_array[i],
          &alpha,
          a_array[idx], lda_array[i],
          b_array[idx], ldb_array[i],
          &beta_array[i],
          c_array[idx], ldc_array[i]);
      ++idx;
    }
  }
#endif
}



} /* hp_numeric */
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H */
