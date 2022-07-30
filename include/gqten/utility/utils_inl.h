// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-11 16:20
*
* Description: GraceQ/tensor project. Inline utility functions used by template headers.
*/
#ifndef GQTEN_UTILITY_UTILS_INL_H
#define GQTEN_UTILITY_UTILS_INL_H


#include "gqten/framework/value_t.h"    // CoorsT, ShapeT
#include "gqten/framework/consts.h"

#include <vector>
#include <numeric>
#include <complex>
#include <cmath>        // abs
#include <algorithm>    // swap

#include <string.h>     // memcpy
#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


//// Algorithms
// Inplace reorder a vector.
template <typename T>
void InplaceReorder(std::vector<T> &v, const std::vector<size_t> &order) {
  std::vector<size_t> indices(order);
  for (size_t i = 0; i < indices.size(); ++i) {
    auto current = i;
    while (i != indices[current]) {
      auto next = indices[current];
      std::swap(v[current], v[next]);
      indices[current] = current;
      current = next;
    }
    indices[current] = current;
 }
}

inline std::vector<int> Reorder(const std::vector<size_t> &v1, const std::vector<int> &order){
  size_t data_size = v1.size();
  std::vector<int> v2; v2.reserve(data_size);
  for (size_t i = 0; i < data_size; i++ ) {
    v2.push_back(v1[order[i]]);
  }
  return v2;
}

// Calculate Cartesian product.
template<typename T>
T CalcCartProd(T v) {
  T s = {{}};
  for (const auto &u : v) {
    T r;
    r.reserve(s.size() * u.size());
    for (const auto &x : s) {
      for (const auto y : u) {
        r.push_back(x);
        r.back().push_back(y);
      }
    }
    s = std::move(r);
  }
  return s;
}

/*
 * Generate all the coordinates in the following order:
 *    (0, 0, 0,....., 0),
 *    (0, 0, 0,....., 1),
 *    (0, 0, 0,....., 2),
 *    ......
 *    (shape[0], shape[1],.....,shape[n-1]-1),
 *    (shape[0], shape[1],.....,shape[n-1])
 */
inline std::vector<CoorsT> GenAllCoors(const ShapeT &shape) {
  std::vector<CoorsT> each_coors(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    each_coors[i].reserve(shape[i]);
    for (size_t j = 0; j < shape[i]; ++j) {
      each_coors[i].push_back(j);
    }
  }
  return CalcCartProd(each_coors);
}


inline std::vector<size_t> CalcMultiDimDataOffsets(const ShapeT &shape) {
  auto ndim = shape.size();
  if (ndim == 0) { return {}; }
  std::vector<size_t> offsets(ndim);
  offsets[ndim - 1] = 1;
  for(int i = ndim-2; i>=0;--i){
    offsets[i]=offsets[i+1]*shape[i+1];
  }
  return offsets;
}


// Calculate offset for the effective one dimension array.
inline size_t CalcEffOneDimArrayOffset(
    const CoorsT &coors,
    const std::vector<size_t> &data_offsets
) {
  assert(coors.size() == data_offsets.size());
  size_t ndim = coors.size();
  size_t offset = 0;
  for (size_t i = 0; i < ndim; ++i) {
    offset += coors[i] * data_offsets[i];
  }
  return offset;
}


// Multiply selected elements in a vector
template <typename T>
inline T VecMultiSelectElemts(
    const std::vector<T> &v,
    const std::vector<size_t> elem_idxes
) {
  auto selected_elem_num = elem_idxes.size();
  if(selected_elem_num == 0){
    return T(1);
  }
  T res;
  if (selected_elem_num == 1) {
    return v[elem_idxes[0]];
  } else {
    res = v[elem_idxes[0]];
  }
  for (size_t i = 1; i < selected_elem_num; ++i) {
    res *= v[elem_idxes[i]];
  }
  return res;
}


// Add two coordinates together
inline CoorsT CoorsAdd(const CoorsT &coors1, const CoorsT &coors2) {
  assert(coors1.size() == coors2.size());
  CoorsT res;
  res.reserve(coors1.size());
  for (size_t i = 0; i < coors1.size(); ++i) {
    res.push_back(coors1[i] + coors2[i]);
  }
  return res;
}


//// Equivalence check
inline bool DoubleEq(const GQTEN_Double a, const GQTEN_Double b) {
  if (std::abs(a-b) < kDoubleEpsilon) {
    return true;
  } else {
    return false;
  }
}


inline bool ComplexEq(const GQTEN_Complex a, const GQTEN_Complex b) {
  if (std::abs(a-b) < kDoubleEpsilon) {
    return true;
  } else {
    return false;
  }
}


inline bool ArrayEq(
    const GQTEN_Double *parray1, const size_t size1,
    const GQTEN_Double *parray2, const size_t size2) {
  if (size1 !=  size2) {
    return false;
  }
  for (size_t i = 0; i < size1; ++i) {
    if (!DoubleEq(parray1[i], parray2[i])) {
      return false;
    }
  }
  return true;
}


inline bool ArrayEq(
    const GQTEN_Complex *parray1, const size_t size1,
    const GQTEN_Complex *parray2, const size_t size2) {
  if (size1 !=  size2) {
    return false;
  }
  for (size_t i = 0; i < size1; ++i) {
    if (!ComplexEq(parray1[i], parray2[i])) {
      return false;
    }
  }
  return true;
}


//// Random
inline GQTEN_Double drand(void) {
  return GQTEN_Double(rand()) / RAND_MAX;
}


inline GQTEN_Complex zrand(void) {
  return GQTEN_Complex(drand(), drand());
}


inline void Rand(GQTEN_Double &d) {
  d = drand();
}


inline void Rand(GQTEN_Complex &z) {
  z = zrand();
}


template <typename ElemType>
inline ElemType RandT() {
  ElemType val;
  Rand(val);
  return val;
}


//// Math
inline GQTEN_Double CalcScalarNorm2(GQTEN_Double d) {
  return d * d;
}


inline GQTEN_Double CalcScalarNorm2(GQTEN_Complex z) {
  return std::norm(z);
}


inline GQTEN_Double CalcScalarNorm(GQTEN_Double d) {
  return std::abs(d);
}


inline GQTEN_Double CalcScalarNorm(GQTEN_Complex z) {
  return std::sqrt(CalcScalarNorm2(z));
}


inline GQTEN_Double CalcConj(GQTEN_Double d) {
  return d;
}


inline GQTEN_Complex CalcConj(GQTEN_Complex z) {
  return std::conj(z);
}


template <typename TenElemType>
inline std::vector<TenElemType> SquareVec(const std::vector<TenElemType> &v) {
  std::vector<TenElemType> res(v.size());
  for (size_t i = 0; i < v.size(); ++i) { res[i] = std::pow(v[i], 2.0); }
  return res;
}


template <typename TenElemType>
inline std::vector<TenElemType> NormVec(const std::vector<TenElemType> &v) {
  TenElemType sum = std::accumulate(v.begin(), v.end(), 0.0);
  std::vector<TenElemType> res(v.size());
  for (size_t i = 0; i < v.size(); ++i) { res[i] = v[i] / sum; }
  return res;
}


//// Matrix operation
template<typename T>
inline std::vector<T> SliceFromBegin(const std::vector<T> &v, size_t to) {
  auto first = v.cbegin();
  return std::vector<T>(first, first+to);
}


template<typename T>
inline std::vector<T> SliceFromEnd(const std::vector<T> &v, size_t to) {
  auto last = v.cend();
  return std::vector<T>(last-to, last);
}


template <typename ElemT>
void SubMatMemCpy(
    const size_t m, const size_t n,
    const size_t row_offset, const size_t col_offset,
    const size_t sub_m, const size_t sub_n,
    const ElemT *sub_mem_begin,
    ElemT *mem_begin
) {
  size_t offset = row_offset * n + col_offset;
  size_t sub_offset = 0;
  for (size_t row_idx = row_offset; row_idx < row_offset + sub_m; ++row_idx) {
    memcpy(
        mem_begin + offset,
        sub_mem_begin + sub_offset,
        sub_n * sizeof(ElemT)
    );
    offset += n;
    sub_offset += sub_n;
  }
}


//template <typename MatElemType>
//inline MatElemType *MatGetRows(
    //const MatElemType *mat, const long &rows, const long &cols,
    //const long &from, const long &num_rows) {
  //auto new_size = num_rows*cols;
  //auto new_mat = new MatElemType [new_size];
  //std::memcpy(new_mat, mat+(from*cols), new_size*sizeof(MatElemType));
  //return new_mat;
//}


//template <typename MatElemType>
//inline void MatGetRows(
    //const MatElemType *mat, const long &rows, const long &cols,
    //const long &from, const long &num_rows,
    //MatElemType *new_mat) {
  //auto new_size = num_rows*cols;
  //std::memcpy(new_mat, mat+(from*cols), new_size*sizeof(MatElemType));
//}


//template <typename MatElemType>
//inline void MatGetCols(
    //const MatElemType *mat, const long rows, const long cols,
    //const long from, const long num_cols,
    //MatElemType *new_mat) {
  //long offset = from;
  //long new_offset = 0;
  //for (long i = 0; i < rows; ++i) {
    //std::memcpy(new_mat+new_offset, mat+offset, num_cols*sizeof(MatElemType));
    //offset += cols;
    //new_offset += num_cols;
  //}
//}


//template <typename MatElemType>
//inline MatElemType *MatGetCols(
    //const MatElemType *mat, const long rows, const long cols,
    //const long from, const long num_cols) {
  //auto new_size = num_cols * rows;
  //auto new_mat = new MatElemType [new_size];
  //MatGetCols(mat, rows, cols, from, num_cols, new_mat);
  //return new_mat;
//}


//inline void GenDiagMat(
    //const double *diag_v, const long &diag_v_dim, double *full_mat) {
  //for (long i = 0; i < diag_v_dim; ++i) {
    //*(full_mat + (i*diag_v_dim + i)) = diag_v[i];
  //}
//}


//// Free the resources of a GQTensor.
//template <typename TenElemType>
//inline void GQTenFree(GQTensor<TenElemType> *pt) {
  //for (auto &pblk : pt->blocks()) { delete pblk; }
//}


} /* gqten */
#endif /* ifndef GQTEN_UTILITY_UTILS_INL_H */
