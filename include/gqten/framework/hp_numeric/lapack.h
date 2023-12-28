// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-02 14:15
* 
* Description: GraceQ/tensor project. High performance LAPACK related functions
* based on MKL.
*/

/**
@file lapack.h
@brief High performance LAPACK related functions based on MKL.
*/
#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H

#include "gqten/framework/value_t.h"
#include "gqten/framework/flops_count.h"  // flop

#include <algorithm>    // min
#include <cstring>      // memcpy, memset

#ifdef Release
#define NDEBUG
#endif

#include <assert.h>     // assert

#ifndef USE_OPENBLAS

#include "mkl.h"      // cblas_*axpy, cblas_*scal

#else

#include <cblas.h>
#include <lapacke.h>

#endif

namespace gqten {

namespace hp_numeric {

inline lapack_int MatSVD(
    GQTEN_Double *mat,
    const size_t m, const size_t n,
    GQTEN_Double *&u,
    GQTEN_Double *&s,
    GQTEN_Double *&vt
) {
  auto lda = n;
  size_t ldu = std::min(m, n);
  size_t ldvt = n;
  u = (GQTEN_Double *) malloc((ldu * m) * sizeof(GQTEN_Double));
  s = (GQTEN_Double *) malloc(ldu * sizeof(GQTEN_Double));
  vt = (GQTEN_Double *) malloc((ldvt * ldu) * sizeof(GQTEN_Double));
#ifdef FAST_SVD
  auto info = LAPACKE_dgesdd(
      LAPACK_ROW_MAJOR, 'S',
      m, n,
      mat, lda,
      s,
      u, ldu,
      vt, ldvt
  );
#else // More stable
  double *superb = new double[m];
  auto info = LAPACKE_dgesvd(
      LAPACK_ROW_MAJOR, 'S', 'S',
      m, n,
      mat, lda,
      s,
      u, ldu,
      vt, ldvt,
      superb
  );
  delete[] superb;
#endif
  assert(info == 0);
#ifdef GQTEN_COUNT_FLOPS
  flop += 4 * m * n * n - 4 * n * n * n / 3;
  // a rough estimation
#endif
  return info;
}

inline lapack_int MatSVD(
    GQTEN_Complex *mat,
    const size_t m, const size_t n,
    GQTEN_Complex *&u,
    GQTEN_Double *&s,
    GQTEN_Complex *&vt
) {

  auto lda = n;
  size_t ldu = std::min(m, n);
  size_t ldvt = n;
  u = (GQTEN_Complex *) malloc((ldu * m) * sizeof(GQTEN_Complex));
  s = (GQTEN_Double *) malloc(ldu * sizeof(GQTEN_Double));
  vt = (GQTEN_Complex *) malloc((ldvt * ldu) * sizeof(GQTEN_Complex));
#ifdef FAST_SVD
  auto info = LAPACKE_zgesdd(
      LAPACK_ROW_MAJOR, 'S',
      m, n,
      reinterpret_cast<lapack_complex_double *>(mat), lda,
      s,
      reinterpret_cast<lapack_complex_double *>(u), ldu,
      reinterpret_cast<lapack_complex_double *>(vt), ldvt
  );
#else // stable
  double *superb = new double[m];
  auto info = LAPACKE_zgesvd(
      LAPACK_ROW_MAJOR, 'S', 'S',
      m, n,
      reinterpret_cast<lapack_complex_double *>(mat), lda,
      s,
      reinterpret_cast<lapack_complex_double *>(u), ldu,
      reinterpret_cast<lapack_complex_double *>(vt), ldvt,
      superb
  );
  delete[] superb;

#endif
  assert(info == 0);
#ifdef GQTEN_COUNT_FLOPS
  flop += 8 * m * n * n - 8 * n * n * n / 3;
  // a rough estimation
#endif
  return info;
}

inline void MatQR(
    GQTEN_Double *mat,
    const size_t m, const size_t n,
    GQTEN_Double *&q,
    GQTEN_Double *&r
) {
  auto k = std::min(m, n);
  size_t elem_type_size = sizeof(GQTEN_Double);
  auto tau = (GQTEN_Double *) malloc(k * elem_type_size);
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, mat, n, tau);

  // Create R matrix
  r = (GQTEN_Double *) malloc((k * n) * elem_type_size);
  for (size_t i = 0; i < k; ++i) {
    memset(r + i * n, 0, i * elem_type_size);
    memcpy(r + i * n + i, mat + i * n + i, (n - i) * elem_type_size);
  }

  // Create Q matrix
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, k, k, mat, n, tau);     // or: orthogonal
  free(tau);
  q = (GQTEN_Double *) malloc((m * k) * elem_type_size);
  if (m == n) {
    memcpy(q, mat, (m * n) * elem_type_size);
  } else {
    for (size_t i = 0; i < m; ++i) {
      memcpy(q + i * k, mat + i * n, k * elem_type_size);
    }
  }
#ifdef GQTEN_COUNT_FLOPS
  flop += 2 * m * n * n - 2 * n * n * n / 3;
  // the book "Numerical Linear Algebra" by Trefethen and Bau
  // assume Householder transformations
#endif
}

inline void MatQR(
    GQTEN_Complex *mat,
    const size_t m, const size_t n,
    GQTEN_Complex *&q,
    GQTEN_Complex *&r
) {
  auto k = std::min(m, n);
  size_t elem_type_size = sizeof(GQTEN_Complex);
  auto tau = (GQTEN_Complex *) malloc(k * elem_type_size);

  LAPACKE_zgeqrf(LAPACK_ROW_MAJOR, m, n,
                 reinterpret_cast<lapack_complex_double *>(mat),
                 n, reinterpret_cast<lapack_complex_double *>(tau));

  // Create R matrix
  r = (GQTEN_Complex *) malloc((k * n) * elem_type_size);
  for (size_t row = 0; row < k; ++row) {
    memset(r + row * n, 0, row * elem_type_size);
    memcpy(r + row * n + row, mat + row * n + row, (n - row) * elem_type_size);
  }

  // Create Q matrix
  LAPACKE_zungqr(LAPACK_ROW_MAJOR, m, k, k,
                 reinterpret_cast<lapack_complex_double *>(mat),
                 n, reinterpret_cast<lapack_complex_double *>(tau));
  free(tau);
  q = (GQTEN_Complex *) malloc((m * k) * elem_type_size);
  if (m == n) {
    memcpy(q, mat, (m * n) * elem_type_size);
  } else {
    for (size_t i = 0; i < m; ++i) {
      memcpy(q + i * k, mat + i * n, k * elem_type_size);
    }
  }
#ifdef GQTEN_COUNT_FLOPS
  flop += 8 * m * n * n - 8 * n * n * n / 3;
  // the book "Numerical Linear Algebra" by Trefethen and Bau
  // assume Householder transformations
  // roughly estimate for complex number
#endif
}
} /* hp_numeric */
} /* gqten */
#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_LAPACK_H */






//void qr( double* const _Q, double* const _R, double* const _A, const size_t _m, const size_t _n) {
//// Maximal rank is used by Lapacke
//const size_t rank = std::min(_m, _n);

//// Tmp Array for Lapacke
//const std::unique_ptr<double[]> tau(new double[rank]);

//// Calculate QR factorisations
//LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (int) _m, (int) _n, _A, (int) _n, tau.get());

//// Copy the upper triangular Matrix R (rank x _n) into position
//for(size_t row =0; row < rank; ++row) {
//memset(_R+row*_n, 0, row*sizeof(double)); // Set starting zeros
//memcpy(_R+row*_n+row, _A+row*_n+row, (_n-row)*sizeof(double)); // Copy upper triangular part from Lapack result.
//}

//// Create orthogonal matrix Q (in tmpA)
//LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (int) _m, (int) rank, (int) rank, _A, (int) _n, tau.get());

////Copy Q (_m x rank) into position
//if(_m == _n) {
//memcpy(_Q, _A, sizeof(double)*(_m*_n));
//} else {
//for(size_t row =0; row < _m; ++row) {
//memcpy(_Q+row*rank, _A+row*_n, sizeof(double)*(rank));
//}
//}
//}
