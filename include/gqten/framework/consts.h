// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-11 20:43
* 
* Description: GraceQ/tensor project. Constants used by this library.
*/
#ifndef GQTEN_FRAMEWORK_CONSTS_H
#define GQTEN_FRAMEWORK_CONSTS_H


#include <string>


namespace gqten {


// GQTensor storage file suffix.
const std::string kGQTenFileSuffix = "gqten";

// Double numerical error.
const double kDoubleEpsilon = 1.0E-15;

// Default tensor transpose threads number.
const int kTensorTransposeDefaultNumThreads = 4;

// Default OpenMP thread number(in every processor)
const unsigned kOmpDefaultTotalNumThreads = 4;

// MPI master's rank
const size_t kMPIMasterRank = 0;
} /* gqten */ 
#endif /* ifndef GQTEN_FRAMEWORK_CONSTS_H */
