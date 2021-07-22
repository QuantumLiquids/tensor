// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author:   Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
            Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-7-22 
*
* Description: GraceQ/tensor project. Fuse two indices of tensor.
*/

/**
@file ten_expand.h
@brief Fuse two indices of tensor.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_FUSE_INDEX_H
#define GQTEN_TENSOR_MANIPULATION_TEN_FUSE_INDEX_H


#include "gqten/gqtensor_all.h"     // GQTensor
#include "gqten/tensor_manipulation/index_combine.h"  // QNSctsOffsetInfo

#include <vector>       // vector
#include <map>          // map
#include <algorithm>    // find

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert



namespace gqten {

/** tensor fuse two indices

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param idx1 the number of index which need to fuse
@param idx2

@note the fused index will be put in first index
*/


template <typename QNT>
Index<QNT> FuseTwoIndexAndRecordInfo(
    const Index<QNT> &idx1,
    const Index<QNT> &idx2,
    std::vector<QNSctsOffsetInfo>& qnscts_offset_info_list
);

template <typename TenElemT, typename QNT>
void GQTensor<TenElemT, QNT>::FuseIndex(
    const size_t idx1, 
    const size_t idx2
){
  assert(idx1<idx2 && idx2<rank_);
  assert(indexes_[idx1].GetDir()==indexes_[idx2].GetDir());
  
  //First we transpose the idx1 and idx2 to last 
  std::vector<size_t> transpose_axes(rank_);
  transpose_axes[0]=idx1;
  transpose_axes[1]=idx2;
  for(size_t i = 2; i < idx1+2 ;i++){
    transpose_axes[i] = i-2;
  }
  for(size_t i = idx1+2; i<idx2+1;i++){
    transpose_axes[i] = i-1;
  }
  for(size_t i = idx2+1;i<rank_;i++){
    transpose_axes[i] = i;
  }
  Transpose(transpose_axes);

  std::vector<QNSctsOffsetInfo> qnscts_offset_info_list;
  auto old_index_0 = indexes_[0];
  auto old_index_1 = indexes_[1];
  Index<QNT> new_index = FuseTwoIndexAndRecordInfo(
    old_index_0, 
    old_index_1,
    qnscts_offset_info_list
    );
  indexes_.erase(indexes_.begin());
  indexes_[0] = new_index;
  shape_[1] = shape_[0]*shape_[1];
  shape_.erase(shape_.begin());

  pblk_spar_data_ten_->FuseFirstTwoIndex(
    old_index_0,
    old_index_1,
    qnscts_offset_info_list
    );
}

template <typename QNT>
Index<QNT> FuseTwoIndexAndRecordInfo(
    const Index<QNT> &idx1,
    const Index<QNT> &idx2,
    std::vector<QNSctsOffsetInfo>& qnscts_offset_info_list
){
  assert(idx1.GetDir() != GQTenIndexDirType::NDIR);
  assert(idx2.GetDir() != GQTenIndexDirType::NDIR);
  // assert(idx1.GetDir()==idx1.GetDir());
  auto new_idx_dir = idx1.GetDir();
  auto idx1_qnsct_num = idx1.GetQNSctNum();
  auto idx2_qnsct_num = idx2.GetQNSctNum();
  qnscts_offset_info_list.reserve(idx1_qnsct_num * idx2_qnsct_num);
  std::vector<QNDgnc<QNT>> new_qn_dgnc_list;
  for (size_t i = 0; i < idx1_qnsct_num; ++i) {
    for (size_t j = 0; j < idx2_qnsct_num; ++j) {
      QNSector<QNT> qnsct_from_idx1 = idx1.GetQNSct(i);
      QNSector<QNT> qnsct_from_idx2 = idx2.GetQNSct(j);
      auto qn_from_idx1 = qnsct_from_idx1.GetQn();
      auto qn_from_idx2 = qnsct_from_idx2.GetQn();
      auto dgnc_from_idx1 = qnsct_from_idx1.GetDegeneracy();
      auto dgnc_from_idx2 = qnsct_from_idx2.GetDegeneracy();
      QNT combined_qn;
      switch (idx1.GetDir()) {
        case GQTenIndexDirType::IN:
          switch (idx2.GetDir()) {
            case GQTenIndexDirType::IN:
              combined_qn = qn_from_idx1 + qn_from_idx2;
              break;
            case GQTenIndexDirType::OUT:
              combined_qn = qn_from_idx1 - qn_from_idx2;
              break;
            default:
              assert(false);
          }
          break;
        case GQTenIndexDirType::OUT:
          switch (idx2.GetDir()) {
            case GQTenIndexDirType::IN:
              combined_qn = qn_from_idx2 - qn_from_idx1;
              break;
            case GQTenIndexDirType::OUT:
              combined_qn = (-qn_from_idx2) - qn_from_idx1;
              break;
            default:
              assert(false);
          }
          break;
        default:
          assert(false);
      }
      if (new_idx_dir == GQTenIndexDirType::IN) {
        combined_qn = -combined_qn;
      }
      auto poss_it = std::find_if(
                         new_qn_dgnc_list.begin(),
                         new_qn_dgnc_list.end(),
                         [&combined_qn](const QNDgnc<QNT> &qn_dgnc)->bool {
                           return qn_dgnc.first == combined_qn;
                         }
                     );
      if (poss_it != new_qn_dgnc_list.end()) {
        size_t offset = poss_it->second;
        poss_it->second += (dgnc_from_idx1 * dgnc_from_idx2);
        size_t qn_dgnc_idx = poss_it - new_qn_dgnc_list.begin();
        qnscts_offset_info_list.emplace_back(
            std::make_tuple(i, j, qn_dgnc_idx, offset)
        );
      } else {
        size_t qn_dgnc_idx = new_qn_dgnc_list.size();
        new_qn_dgnc_list.push_back(
            std::make_pair(combined_qn, dgnc_from_idx1 * dgnc_from_idx2)
        );
        qnscts_offset_info_list.emplace_back(
            std::make_tuple(i, j, qn_dgnc_idx, 0)
        );
      }
    }
  }
  QNSectorVec<QNT> qnscts;
  // std::vector<size_t> new_idx_qnsct_dim_offsets;
  qnscts.reserve(new_qn_dgnc_list.size());
  // new_idx_qnsct_dim_offsets.reserve(new_qn_dgnc_list.size());
  // size_t qnsct_dim_offset = 0;
  for (auto &new_qn_dgnc : new_qn_dgnc_list) {
    qnscts.push_back(QNSector<QNT>(new_qn_dgnc.first, new_qn_dgnc.second));
    // new_idx_qnsct_dim_offsets.push_back(qnsct_dim_offset);
    // qnsct_dim_offset += new_qn_dgnc.second;
  }
  return Index<QNT>(qnscts, new_idx_dir);
}
}

#endif //GQTEN_TENSOR_MANIPULATION_TEN_FUSE_INDEX_H