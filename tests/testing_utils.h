// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-29 16:41
* 
* Description: GraceQ/tensor project. Testing utilities.
*/
#ifndef GQTEN_TESTING_UTILS_H
#define GQTEN_TESTING_UTILS_H


#include <vector>

#include "gtest/gtest.h"
#include "gqten/detail/value_t.h"


using namespace gqten;


inline std::vector<long> TransCoors(
    const std::vector<long> &old_coors, const std::vector<long> &axes_map) {
  std::vector<long> new_coors(old_coors.size());
  for (std::size_t i = 0; i < axes_map.size(); ++i) {
    new_coors[i] = old_coors[axes_map[i]];
  }
  return new_coors;
}


inline void EXPECT_COMPLEX_EQ(
    const GQTEN_Complex &lhs,
    const GQTEN_Complex &rhs) {
  EXPECT_DOUBLE_EQ(lhs.real(), rhs.real());
  EXPECT_DOUBLE_EQ(lhs.imag(), rhs.imag());
}


inline void GtestArrayEq(const double *lhs, const double *rhs, const long len) {
  for (long i = 0; i < len; ++i) {
    EXPECT_DOUBLE_EQ(lhs[i], rhs[i]);
  }
}


inline void GtestArrayEq(
    const GQTEN_Complex *lhs, const GQTEN_Complex *rhs, const long len) {
  for (long i = 0; i < len; ++i) {
    EXPECT_COMPLEX_EQ(lhs[i], rhs[i]);
  }
}
#endif /* ifndef GQTEN_TESTING_UTILS_H */
