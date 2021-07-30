// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang
* Creation Date: 2021-07-30
*
* Description: GraceQ/tensor project. Expand two tensors, magic changing version.
*/

/**
@file ten_mc_expand.h
@brief Expand two tensors, magic change version, which means only add new quantum sectors in the identified index of seconde tensor.
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_MC_EXPAND_H
#define GQTEN_TENSOR_MANIPULATION_TEN_MC_EXPAND_H


#include "gqten/gqtensor_all.h"     // GQTensor
#include "gqten/utility/timer.h"
#include "gqten/tensor_manipulation/ten_expand.h" //Checker

#include <vector>       // vector
#include <map>          // map
#include <algorithm>    // find

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {
// Forward declaration
template <typename TenElemT, typename QNT>
void ExpandMCOneIdx_(
    GQTensor<TenElemT, QNT> *,
    GQTensor<TenElemT, QNT> *,
    const size_t,
    GQTensor<TenElemT, QNT> *
);

/**
Function version for tensor magic changing expansion.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param cpa Pointer to input tensor \f$ A \f$.
@param cpb Pointer to input tensor \f$ B \f$.
@param expand_idx_nums Index numbers which index to be expanded.
@param pc Pointer to result tensor \f$ C \f$.

@note cpa and cpb will actually be temporarily changed in this function. This
      defect will be fixed in the future!
*/
template <typename TenElemT, typename QNT>
void ExpandMC(
    const GQTensor<TenElemT, QNT> *cpa,
    const GQTensor<TenElemT, QNT> *cpb,
    const std::vector<size_t> &expand_idx_nums,
    GQTensor<TenElemT, QNT> *pc
) {
#ifndef NDEBUG
  TensorExpandPreChecker(*cpa, *cpb, expand_idx_nums);
#endif /* ifndef NDEBUG */

  // TODO: Remove const_cast!!
  auto pa = const_cast<GQTensor<TenElemT, QNT> *>(cpa);
  auto pb = const_cast<GQTensor<TenElemT, QNT> *>(cpb);

  if (expand_idx_nums.size() == 1) {
    ExpandMCOneIdx_(pa, pb, expand_idx_nums[0], pc);
    return;
  } else {
    std::cout << "Sorry that magic version expansion on multi-index is not support now." <<std::endl;
    exit(0);
    /*
    std::vector<size_t> expand_idx_nums_without_last_one(
        expand_idx_nums.begin(), expand_idx_nums.end() - 1
    );
    auto indexes_a = pa->GetIndexes();
    auto indexes_b = pb->GetIndexes();
    auto expand_dual_a_indexes = pa->GetIndexes();
    auto expand_dual_b_indexes = pb->GetIndexes();
    for(auto idx_num : expand_idx_nums_without_last_one){
      expand_dual_a_indexes[idx_num] = indexes_b[idx_num];
      expand_dual_b_indexes[idx_num] = indexes_a[idx_num];
    }

    GQTensor<TenElemT, QNT> expand_tmp_a, expand_tmp_b;
    {
      GQTensor<TenElemT, QNT> expand_dual_a(expand_dual_a_indexes);
      Expand(
          pa, &expand_dual_a,
          expand_idx_nums_without_last_one,
          &expand_tmp_a
      );
    }
    {
      GQTensor<TenElemT, QNT> expand_dual_b(expand_dual_b_indexes);
      Expand(
          &expand_dual_b, pb,
          expand_idx_nums_without_last_one,
          &expand_tmp_b
      );
    }

    ExpandOneIdx_(&expand_tmp_a, &expand_tmp_b, expand_idx_nums.back(), pc);
    */
  }

#ifndef NDEBUG
  ExpandedTenDivChecker(*pa, *pb, *pc);
#endif /* ifndef NDEBUG */
}


template <typename QNT>
inline Index<QNT> ExpandIndexMCAndRecordInfo(
    const Index<QNT> &idx_from_a,
    const Index<QNT> &idx_from_b,
    std::map<size_t, int> &b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
) {
  QNSectorVec<QNT> expanded_qnscts = idx_from_a.GetQNScts();
  auto qnscts_from_b = idx_from_b.GetQNScts();
  auto qnscts_from_a_size = idx_from_a.GetQNSctNum();
  auto qnscts_from_b_size = idx_from_b.GetQNSctNum();
  expanded_qnscts.reserve(qnscts_from_a_size+qnscts_from_b_size);
  for(size_t i=0;i<qnscts_from_b_size;i++){
    QNSector<QNT>& qnsct_in_b = qnscts_from_b[i];
    auto iter = std::find_if(expanded_qnscts.cbegin(), 
                expanded_qnscts.cbegin()+qnscts_from_a_size,
                [qnsct_in_b](QNSector<QNT> qnsct_in_a){
                  return qnsct_in_a.GetQn() == qnsct_in_b.GetQn();
                } );
    if(iter == expanded_qnscts.cbegin()+qnscts_from_a_size){//didn't find
      b_idx_qnsct_coor_expanded_idx_qnsct_coor_map.insert(
                      std::make_pair( i, expanded_qnscts.size() )
                      );
      expanded_qnscts.push_back(qnsct_in_b);
    }else{//found
      b_idx_qnsct_coor_expanded_idx_qnsct_coor_map.insert(
              std::make_pair(i, -1)
      );
    }
  }
  return Index<QNT>(expanded_qnscts, idx_from_a.GetDir());
}


/**
Tensor expansion: special case for only expand on one index.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param expand_idx_num Index number which index to be expanded.
@param pc Pointer to result tensor \f$ C \f$.

@note This function will temporarily modify two input tensors! Intra-used
      function, not for normal user!
*/
template <typename TenElemT, typename QNT>
void ExpandMCOneIdx_(
    GQTensor<TenElemT, QNT> *pa,
    GQTensor<TenElemT, QNT> *pb,
    const size_t expand_idx_num,
    GQTensor<TenElemT, QNT> *pc
) {
#ifndef NDEBUG
  TensorExpandPreChecker(*pa, *pb, {expand_idx_num});
#endif /* ifndef NDEBUG */

#ifdef GQTEN_TIMING_MODE
  Timer expand_pre_transpose_timer("   =============> expansion_first_time_transpose");
#endif
  // Firstly we transpose the expand_idx_num-th index to the first index
  size_t ten_rank = pa->Rank();
  std::vector<size_t> transpose_order(ten_rank);
  transpose_order[0] = expand_idx_num;
  for(size_t i = 1; i <= expand_idx_num; i++){ transpose_order[i] = i - 1; }
  for(size_t i = expand_idx_num+1; i < ten_rank; i++){ transpose_order[i] = i; }
  if (pa == pb) {
    pa->Transpose(transpose_order);
  } else {
    pa->Transpose(transpose_order);
    pb->Transpose(transpose_order);
  }
#ifdef GQTEN_TIMING_MODE
  expand_pre_transpose_timer.PrintElapsed();
#endif

#ifdef GQTEN_TIMING_MODE
  Timer expand_index_timer("   =============> expansion_expand_index_and_record_info");
#endif
  // Then we can expand the two tensor according the first indexes. For each block, the data are direct connected
  // Expand the first index
  std::map<size_t, int> b_idx_qnsct_coor_expanded_idx_qnsct_coor_map;
  auto expanded_index = ExpandIndexMCAndRecordInfo(
      pa->GetIndexes()[0],
      pb->GetIndexes()[0],
      b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
  );
#ifdef GQTEN_TIMING_MODE
  expand_index_timer.PrintElapsed();
#endif 
  // Expand data
  IndexVec<QNT> expanded_idxs = pa->GetIndexes();
  expanded_idxs[0] = expanded_index;
  (*pc) = GQTensor<TenElemT, QNT>(expanded_idxs);
  (pc->GetBlkSparDataTen()).ConstructMCExpandedDataOnFirstIndex(
      pa->GetBlkSparDataTen(),
      pb->GetBlkSparDataTen(),
      b_idx_qnsct_coor_expanded_idx_qnsct_coor_map
  );
#ifdef GQTEN_TIMING_MODE
  Timer expand_latter_transpose_timer("   =============> expansion_second_time_transpose");
#endif
  // transpose back
  for(size_t i = 0; i < expand_idx_num; i++){ transpose_order[i] = i + 1; }
  transpose_order[expand_idx_num] = 0;
  if (pa == pb) {
    pa->Transpose(transpose_order);
  } else {
    pa->Transpose(transpose_order);
    pb->Transpose(transpose_order);
  }
  pc->Transpose(transpose_order);
#ifdef GQTEN_TIMING_MODE
  expand_latter_transpose_timer.PrintElapsed();
#endif
#ifndef NDEBUG
  ExpandedTenDivChecker(*pa, *pb, *pc);
#endif /* ifndef NDEBUG */
}
} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_TEN_EXPAND_H */
