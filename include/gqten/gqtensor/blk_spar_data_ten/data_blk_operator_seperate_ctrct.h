// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-7-29
*
* Description: GraceQ/tensor project. Data block level operations for block
* sparse data tensor for seperated contract task.
*/

#pragma once


#include "gqten/gqtensor/blk_spar_data_ten/blk_spar_data_ten.h"
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operations.h"
#include "gqten/gqtensor/blk_spar_data_ten/raw_data_operation_tasks.h"
#include "gqten/framework/hp_numeric/lapack.h"    // MatSVD, MatQR
#include "gqten/utility/timer.h"
#include "gqten/gqtensor/blk_spar_data_ten/data_blk_operations.h"

namespace gqten{

/**
Generate data blocks for two tensor contraction.

@param bsdt_a Block sparse data tensor A.
@param bsdt_b Block sparse data tensor B.
@param ctrct_axes_set To-be contracted tensor axes indexes.
       For example, {{0, 1}, {3, 2}}.
*/
template <typename ElemT, typename QNT>
std::vector<RawDataCtrctTask>
BlockSparseDataTensor<ElemT, QNT>::DataBlkGenForTenCtrct(
    const BlockSparseDataTensor &bsdt_a,
    const size_t idx_a,
    const size_t qn_sector_idx_a,
    const BlockSparseDataTensor &bsdt_b,
    const size_t idx_b,
    const size_t qn_sector_idx_b,
    const std::vector<std::vector<size_t>> &ctrct_axes_set
) {
  assert(!(bsdt_a.IsScalar() || bsdt_b.IsScalar()));
  auto saved_axes_set = TenCtrctGenSavedAxesSet(
                            bsdt_a.ten_rank,
                            bsdt_b.ten_rank,
                            ctrct_axes_set
                        );
  std::vector<size_t> a_trans_orders, b_trans_orders;
  auto a_b_need_trans = TenCtrctNeedTransCheck(
                            ctrct_axes_set,
                            saved_axes_set,
                            a_trans_orders,
                            b_trans_orders
                        );
  std::__1::map<size_t, gqten::DataBlk<QNT>> a_blk_idx_data_blk_map = bsdt_a.GetBlkIdxDataBlkMap();
  std::__1::map<size_t, gqten::DataBlk<QNT>> b_blk_idx_data_blk_map = bsdt_b.GetBlkIdxDataBlkMap();
  if(idx_a < bsdt_a.ten_rank ){
    for(auto iter = a_blk_idx_data_blk_map.begin();
             iter != a_blk_idx_data_blk_map.end();){
      DataBlk<QNT>& data_blk = iter->second;
      if( data_blk.blk_coors[idx_a] != qn_sector_idx_a ){
        a_blk_idx_data_blk_map.erase(iter++);
      }else{
        iter++;
      }
    }
  }
  if(idx_b < bsdt_b.ten_rank ){
    for(auto iter = b_blk_idx_data_blk_map.begin();
             iter != b_blk_idx_data_blk_map.end();){
      DataBlk<QNT> data_blk = iter->second;
      if( data_blk.blk_coors[idx_b] != qn_sector_idx_b ){
        b_blk_idx_data_blk_map.erase(iter++);
      }else{
        iter++;
      }
    }
  }
  auto a_blk_idx_qnblk_info_part_hash_map = GenBlkIdxQNBlkInfoPartHashMap(
                                                a_blk_idx_data_blk_map,
                                                ctrct_axes_set[0]
                                            );
  auto b_blk_idx_qnblk_info_part_hash_map = GenBlkIdxQNBlkInfoPartHashMap(
                                                b_blk_idx_data_blk_map,
                                                ctrct_axes_set[1]
                                            );
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks;
  std::unordered_map<size_t, size_t>
      a_blk_idx_m_map,
      a_blk_idx_k_map,
      b_blk_idx_n_map;
  bool c_is_scalar = IsScalar();
  bool scalar_c_first_task = true;
#ifndef NDEBUG
  if (c_is_scalar) {
    assert(saved_axes_set[0].empty() && saved_axes_set[1].empty());
  }
#endif /* ifndef NDEBUG */
  for (auto &a_blk_idx_part_hash : a_blk_idx_qnblk_info_part_hash_map) {
    for (auto &b_blk_idx_part_hash : b_blk_idx_qnblk_info_part_hash_map) {
      if (a_blk_idx_part_hash.second == b_blk_idx_part_hash.second) {
        auto a_blk_idx = a_blk_idx_part_hash.first;
        auto b_blk_idx = b_blk_idx_part_hash.first;
        auto a_data_blk = a_blk_idx_data_blk_map[a_blk_idx];
        auto b_data_blk = b_blk_idx_data_blk_map[b_blk_idx];
        // Calculate m, k, n
        size_t m, k, n;
        if (c_is_scalar) {
          m = 1;
          n = 1;
        } else {
          if (a_blk_idx_m_map.find(a_blk_idx) != a_blk_idx_m_map.end()) {
            m = a_blk_idx_m_map.at(a_blk_idx);
          } else {
            m = VecMultiSelectElemts(a_data_blk.shape, saved_axes_set[0]);
            a_blk_idx_m_map[a_blk_idx] = m;
          }
          if (b_blk_idx_n_map.find(b_blk_idx) != b_blk_idx_n_map.end()) {
            n = b_blk_idx_n_map.at(b_blk_idx);
          } else {
            n = VecMultiSelectElemts(b_data_blk.shape, saved_axes_set[1]);
            b_blk_idx_n_map[b_blk_idx] = n;
          }
        }
        if (a_blk_idx_k_map.find(a_blk_idx) != a_blk_idx_k_map.end()) {
          k = a_blk_idx_k_map.at(a_blk_idx);
        } else {
          k = VecMultiSelectElemts(a_data_blk.shape, ctrct_axes_set[0]);
          a_blk_idx_k_map[a_blk_idx] = k;
        }

        // Create raw data contraction task
        if (c_is_scalar) {
          GQTEN_Double beta;
          if (scalar_c_first_task) {
            beta = 0.0;
            raw_data_size_ = 1;     // Set raw data size at first task scheduling
            scalar_c_first_task = false;
          } else {
            beta = 1.0;
          }
          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  a_b_need_trans.first,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  a_b_need_trans.second,
                  m, k, n,
                  beta
              )
          );
        } else {
          auto c_blk_coors = GenTenCtrctDataBlkCoors(
                                 a_data_blk.blk_coors,
                                 b_data_blk.blk_coors,
                                 saved_axes_set
                             );
          auto c_blk_idx = BlkCoorsToBlkIdx(c_blk_coors);
          GQTEN_Double beta;
          if (blk_idx_data_blk_map_.find(c_blk_idx) !=
              blk_idx_data_blk_map_.end()
          ) {
            beta = 1.0;
          } else {
            DataBlkInsert(c_blk_coors, false);
            beta = 0.0;
          }
          raw_data_ctrct_tasks.push_back(
              RawDataCtrctTask(
                  a_blk_idx,
                  a_data_blk.data_offset,
                  a_b_need_trans.first,
                  b_blk_idx,
                  b_data_blk.data_offset,
                  a_b_need_trans.second,
                  c_blk_idx,
                  m, k, n,
                  beta
              )
          );
        }
      }
    }
  }

  for (auto &task : raw_data_ctrct_tasks) {
    if (a_b_need_trans.first) {
      task.a_trans_orders = a_trans_orders;
    }
    if (a_b_need_trans.second) {
      task.b_trans_orders = b_trans_orders;
    }
    if (!c_is_scalar) {
      task.c_data_offset = blk_idx_data_blk_map_[task.c_blk_idx].data_offset;
    } else {
      task.c_data_offset = 0;
    }
  }

  return raw_data_ctrct_tasks;
}




/**
Generate data blocks for two tensor contraction.

@param bsdt_a Block sparse data tensor A.
@param bsdt_b Block sparse data tensor B.
@param ctrct_axes_a To-be contracted tensor axes indexes in A
@param ctrct_axes_b To-be contracted tensor axes indexes in B

This arguement list has the same meaning with the other same name function.
The difference of this function is here only generate the DataBlk for tensor contraction.
*/  /*
template <typename ElemT, typename QNT>
void BlockSparseDataTensor<ElemT, QNT>::DataBlkGenForTenCtrct(
    const BlockSparseDataTensor &bsdt_a,
    const BlockSparseDataTensor &bsdt_b,
    const std::vector<size_t> &ctrct_axes_a,
    const std::vector<size_t> &ctrct_axes_b
) {
    assert(!(bsdt_a.IsScalar() || bsdt_b.IsScalar()));
    auto saved_axes_set = TenCtrctGenSavedAxesSet(
                            bsdt_a.ten_rank,
                            bsdt_b.ten_rank,
                            {ctrct_axes_a, ctrct_axes_b}
                        );
    auto a_b_need_trans = TenCtrctNeedTransCheck(
                            ctrct_axes_set,
                            saved_axes_set,
                            a_trans_orders,
                            b_trans_orders
                        );
    auto a_blk_idx_data_blk_map = bsdt_a.GetBlkIdxDataBlkMap();
    auto b_blk_idx_data_blk_map = bsdt_b.GetBlkIdxDataBlkMap();
    auto a_blk_idx_qnblk_info_part_hash_map = GenBlkIdxQNBlkInfoPartHashMap(
                                                a_blk_idx_data_blk_map,
                                                ctrct_axes_set[0]
                                            );
    auto b_blk_idx_qnblk_info_part_hash_map = GenBlkIdxQNBlkInfoPartHashMap(
                                                b_blk_idx_data_blk_map,
                                                ctrct_axes_set[1]
                                            );
    


}



*/
}