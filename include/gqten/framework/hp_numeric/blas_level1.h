// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-24 20:13
*
* Description: GraceQ/tensor project. High performance BLAS Level 1 related
* functions based on MKL.
*/

/**
@file blas_level1.h
@brief High performance BLAS Level 1 related functions based on MKL.
*/
#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H


#include "gqten/framework/value_t.h"      // GQTEN_Double, GQTEN_Complex

#include <string.h>     // memcpy
#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert

#include "mkl.h"      // cblas_*axpy, cblas_*scal


namespace gqten {


/// High performance numerical functions.
namespace hp_numeric {


inline void VectorAddTo(
    const GQTEN_Double *x,
    const size_t size,
    GQTEN_Double *y,
    const GQTEN_Double a = 1.0
) {
  cblas_daxpy(size, a, x, 1, y, 1);
}


inline void VectorAddTo(
    const GQTEN_Complex *x,
    const size_t size,
    GQTEN_Complex *y,
    const GQTEN_Complex a = 1.0
) {
  cblas_zaxpy(size, &a, x, 1, y, 1);
}


inline void VectorScaleCopy(
    const GQTEN_Double *x,
    const size_t size,
    GQTEN_Double *y,
    const GQTEN_Double a = 1.0
) {
  cblas_dcopy(size, x, 1, y, 1);
  cblas_dscal(size, a, y, 1);
}


inline void VectorScaleCopy(
    const GQTEN_Complex *x,
    const size_t size,
    GQTEN_Complex *y,
    const GQTEN_Complex a = 1.0
) {
  cblas_zcopy(size, x, 1, y, 1);
  cblas_zscal(size, &a, y, 1);
}

inline void VectorCopy(
  const GQTEN_Double* source,
  const size_t size,
  GQTEN_Double* dest
){
  cblas_dcopy(size, source, 1, dest, 1);
}

inline void VectorCopy(
  const GQTEN_Complex* source,
  const size_t size,
  GQTEN_Complex* dest
){
  cblas_zcopy(size, source, 1, dest, 1);
}

inline void VectorScale(
    GQTEN_Double *x,
    const size_t size,
    const GQTEN_Double a
) {
  cblas_dscal(size, a, x, 1);
}


inline void VectorScale(
    GQTEN_Complex *x,
    const size_t size,
    const GQTEN_Complex a
) {
  cblas_zscal(size, &a, x, 1);
}

/**
 * @note return sqrt(sum(x^2)) not sum(x^2)
 */
inline double Vector2Norm(
  GQTEN_Double *x,
  const size_t size
){
  return cblas_dnrm2(size, x, 1);
}

inline double Vector2Norm(
  GQTEN_Complex *x,
  const size_t size
){
  return cblas_dznrm2(size, x, 1);
}

inline void VectorRealToCplx(
    const GQTEN_Double *real,
    const size_t size,
    GQTEN_Complex *cplx
) {
  for (size_t i = 0; i < size; ++i) { cplx[i]= real[i]; }
}
} /* hp_numeric */
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_BLAS_LEVEL1_H */
