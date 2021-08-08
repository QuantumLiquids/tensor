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
    const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
    const GQTEN_Double *beta_array,
    GQTEN_Double **c_array, 
    const MKL_INT group_count) {

    const CBLAS_LAYOUT Layout = CblasRowMajor;
    CBLAS_TRANSPOSE* transa_array = new CBLAS_TRANSPOSE[group_count];
    CBLAS_TRANSPOSE* transb_array = new CBLAS_TRANSPOSE[group_count];
    for (MKL_INT i = 0 ; i < group_count ; i++){
      transa_array[i] = CblasNoTrans;
    }
    for (MKL_INT i = 0 ; i < group_count ; i++){
      transb_array[i] = CblasNoTrans;
    }

    GQTEN_Double *alpha_array = new GQTEN_Double[group_count];
    for (MKL_INT i = 0 ; i < group_count ; i++){
      alpha_array[i] = 1.0;
    }

    const MKL_INT* lda_array = k_array;
    const MKL_INT *ldb_array = n_array;
    const MKL_INT *ldc_array = n_array;
    MKL_INT* group_size = new MKL_INT[group_count];
    for (MKL_INT i = 0 ; i < group_count ; i++){
      group_size[i] = 1;
    }
#ifdef GQTEN_USE_MKL_GEMM_BATCH

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

#else // Use direct gemm loop.

  auto idx = 0;
  for (MKL_INT i = 0; i < group_count; ++i) {
    for (MKL_INT j = 0; j < group_size[i]; ++j) {
      cblas_dgemm(
          Layout,
          transa_array[i], transb_array[i],
          m_array[i], n_array[i], k_array[i],
          alpha_array[i],
          a_array[idx], lda_array[i],
          b_array[idx], ldb_array[i],
          beta_array[i],
          c_array[idx], ldc_array[i]);
      ++idx;
    }
  }

#endif
  delete[] transa_array;
  delete[] transb_array;
  delete[] alpha_array;
  delete[] group_size;
}


inline void MatMultiplyBatch(
    const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
    const GQTEN_Complex **a_array, 
    const GQTEN_Complex **b_array, 
    const GQTEN_Complex *beta_array,
    GQTEN_Complex **c_array,
    const MKL_INT group_count) {
    
    const CBLAS_LAYOUT Layout = CblasRowMajor;
    CBLAS_TRANSPOSE* transa_array = new CBLAS_TRANSPOSE[group_count];
    CBLAS_TRANSPOSE* transb_array = new CBLAS_TRANSPOSE[group_count];
    for (MKL_INT i = 0 ; i < group_count ; i++){
      transa_array[i] = CblasNoTrans;
    }
    for (MKL_INT i = 0 ; i < group_count ; i++){
      transb_array[i] = CblasNoTrans;
    }

    GQTEN_Complex *alpha_array = new GQTEN_Complex[group_count];
    for (MKL_INT i = 0 ; i < group_count ; i++){
      alpha_array[i] = GQTEN_Complex(1.0);
    }

    const MKL_INT* lda_array = k_array;
    const MKL_INT *ldb_array = n_array;
    const MKL_INT *ldc_array = n_array;
    MKL_INT* group_size = new MKL_INT[group_count];
    for (MKL_INT i = 0 ; i < group_count ; i++){
      group_size[i] = 1;
    }

#ifdef GQTEN_USE_MKL_GEMM_BATCH

  cblas_zgemm_batch (
      Layout,
      transa_array, transb_array,
      m_array, n_array, k_array,
      &alpha_array,
      a_array, lda_array,
      b_array, ldb_array,
      &beta_array,
      c_array, ldc_array,
      group_count,
      group_size);

#else // Use direct gemm loop.

  auto idx = 0;
  for (MKL_INT i = 0; i < group_count; ++i) {
    for (MKL_INT j = 0; j < group_size[i]; ++j) {
      cblas_zgemm(
          Layout,
          transa_array[i], transb_array[i],
          m_array[i], n_array[i], k_array[i],
          &alpha_array[i],
          a_array[idx], lda_array[i],
          b_array[idx], ldb_array[i],
          &beta_array[i],
          c_array[idx], ldc_array[i]);
      ++idx;
    }
  }
#endif
  delete[] transa_array;
  delete[] transb_array;
  delete[] alpha_array;
  delete[] group_size;
}



} /* hp_numeric */
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL3_H */
