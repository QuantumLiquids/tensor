// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-16 13:05
*
* Description: GraceQ/tensor project. Information of a quantum number block.
*/

/**
@file qnblk_info.h
@brief Information of a quantum number block.
*/
#ifndef GQTEN_GQTENSOR_QNBLK_INFO_H
#define GQTEN_GQTENSOR_QNBLK_INFO_H


#include "gqten/framework/value_t.h"    // CoorsT, ShapeT
#include "gqten/framework/vec_hash.h"   // VecPtrHasher
#include "gqten/gqtensor/qnsct.h"       // QNSectorVec
#include "gqten/utility/utils_inl.h"    // Reorder


namespace gqten {


/**
Information of a quantum number block.

@tparam QNT Type of the quantum number.
*/
template <typename QNT>
class QNBlkInfo {
public:
  QNBlkInfo(void) = default;

  QNBlkInfo(const QNSectorVec<QNT> &qnscts) : qnscts(qnscts) {}

  /**
  Calculate the shape of the degeneracy space.
  */
  ShapeT CalcDgncSpaceShape(void) {
    ShapeT dgnc_space_shape;
    dgnc_space_shape.reserve(qnscts.size());
    for (auto &qnsct : qnscts) {
      dgnc_space_shape.push_back(qnsct.GetDegeneracy());
    }
    return dgnc_space_shape;
  }

  size_t PartHash(const std::vector<size_t> &) const;
  size_t QnHash(void) const;

  /**
  Transpose quantum number sectors.
  */
  void Transpose(const std::vector<size_t> &transed_idxes_order) {
    InplaceReorder(qnscts, transed_idxes_order);
  }

  QNSectorVec<QNT> qnscts;
};


/**
Calculate part hash value using selected axes indexes.


*/
template <typename QNT>
size_t QNBlkInfo<QNT>::PartHash(const std::vector<size_t> &axes) const {
  return VecPartHasher(qnscts, axes);
}


/**
Calculate a hash value only based on quantum numbers but the shape of the degeneracy space.
*/
template <typename QNT>
size_t QNBlkInfo<QNT>::QnHash(void) const {
  std::vector<QNT> qns;
  qns.reserve( qnscts.size() );
  for (auto &qnsct : qnscts) { qns.push_back(qnsct.GetQn()); }
  return VecHasher(qns);
}
} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_QNBLK_INFO_H */
