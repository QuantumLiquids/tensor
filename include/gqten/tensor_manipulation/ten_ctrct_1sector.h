// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-7-29
*
* Description: GraceQ/tensor project. Contract two tensors, one outer index are restrict on only one qn sector
*/

/**
@file ten_ctrct_1sector.h
@brief Contract two tensors, one outer index are restrict on only one qn sector
*/
#ifndef GQTEN_TENSOR_MANIPULATION_TEN_CTRCT_ONE_SECTOR_H
#define GQTEN_TENSOR_MANIPULATION_TEN_CTRCT_ONE_SECTOR_H


#include "gqten/framework/bases/executor.h"                 // Executor
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/basic_operations.h"     // ToComplex
#include "gqten/gqtensor/blk_spar_data_ten/data_blk_operator_seperate_ctrct.h"

#include <vector>     // vector

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


/// Forward declarations. 
/// TenCtrctInitResTen only gives the Indexes of GQTensor *pc,
/// pointer to BSDT = nullptr.
template <typename TenElemT, typename QNT>
void TenCtrctInitResTen(
    const GQTensor<TenElemT, QNT> *,
    const GQTensor<TenElemT, QNT> *,
    const std::vector<std::vector<size_t>> &,
    GQTensor<TenElemT, QNT> *
);


/**
Tensor contraction executor.

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.
*/
template <typename TenElemT, typename QNT>
class TensorContraction1SectorExecutor : public Executor {
public:
  TensorContraction1SectorExecutor(
      const GQTensor<TenElemT, QNT> * pa,
      const size_t idx_a,
      const size_t qn_sector_idx_a,
      const GQTensor<TenElemT, QNT> * pb,
      const std::vector<std::vector<size_t>> &,
      GQTensor<TenElemT, QNT> * pc
  );// split the index of pa->GetIndexes[idx_a], contract restrict on qn_sector_idx_a's qnsector

  TensorContraction1SectorExecutor(
    const GQTensor<TenElemT, QNT> * pa,
    const GQTensor<TenElemT, QNT> * pb,
    const size_t idx_b,
    const size_t qn_sector_idx_b,
    const std::vector<std::vector<size_t>> &,
    GQTensor<TenElemT, QNT> * pc
  );// split the index of pb->GetIndexes[idx_b], contract restrict on qn_sector_idx_b's qnsector

  void Execute(void) override;

private:
  const GQTensor<TenElemT, QNT> *pa_;
  const size_t idx_a_;
  const size_t qn_sector_idx_a_;
  const GQTensor<TenElemT, QNT> *pb_;
  const size_t idx_b_;
  const size_t qn_sector_idx_b_;
  GQTensor<TenElemT, QNT> *pc_;
  const std::vector<std::vector<size_t>> &axes_set_;
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks_;
};


/**
Initialize a tensor contraction executor.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template <typename TenElemT, typename QNT>
TensorContraction1SectorExecutor<TenElemT, QNT>::TensorContraction1SectorExecutor(
    const GQTensor<TenElemT, QNT> *pa,
    const size_t idx_a,
    const size_t qn_sector_idx_a,
    const GQTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<TenElemT, QNT> *pc
) : pa_(pa), idx_a_(idx_a), qn_sector_idx_a_(qn_sector_idx_a),
    pb_(pb), idx_b_(pb->Rank()+10), qn_sector_idx_b_(0),
    pc_(pc), axes_set_(axes_set)
   {
  assert(pc_->IsDefault());    // Only empty tensor can take the result
#ifndef NDEBUG
  // Check indexes matching
  auto indexesa = pa->GetIndexes();
  auto indexesb = pb->GetIndexes();
  for(size_t i = 0; i < axes_set[0].size(); ++i){
    assert(indexesa[axes_set[0][i]] == InverseIndex(indexesb[axes_set[1][i]]));
  }
  // Check if idx_a_ is in the outer legs
  auto iter = find(axes_set[0].begin(),axes_set[0].end(), idx_a);
  assert(iter == axes_set[0].cend());
  // Check if qn_sector_idx_a_ < number of quantum number sector
  assert(qn_sector_idx_a_ < pa->GetIndexes()[idx_a_].GetQNSctNum());
#endif
  TenCtrctInitResTen(pa_, pb_, axes_set_, pc_);
  //Then we assign the DataBlk and and contract tasks
  raw_data_ctrct_tasks_ = pc_->GetBlkSparDataTen().DataBlkGenForTenCtrct(
                              pa_->GetBlkSparDataTen(),
                              idx_a_,
                              qn_sector_idx_a_,
                              pb_->GetBlkSparDataTen(),
                              idx_b_,
                              qn_sector_idx_b_,
                              axes_set_
                          );
  SetStatus(ExecutorStatus::INITED);
}


/**
Initialize a tensor contraction executor.

@param pa Pointer to input tensor \f$ A \f$.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template <typename TenElemT, typename QNT>
TensorContraction1SectorExecutor<TenElemT, QNT>::TensorContraction1SectorExecutor(
    const GQTensor<TenElemT, QNT> *pa,
    const GQTensor<TenElemT, QNT> *pb,
    const size_t idx_b,
    const size_t qn_sector_idx_b,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<TenElemT, QNT> *pc
) : pa_(pa), idx_a_(pa->Rank()+10), qn_sector_idx_a_(0),
    pb_(pb), idx_b_(idx_b), qn_sector_idx_b_(qn_sector_idx_b),
    pc_(pc), axes_set_(axes_set)
   {
  assert(pc_->IsDefault());    // Only empty tensor can take the result
#ifndef NDEBUG
  // Check indexes matching
  auto indexesa = pa->GetIndexes();
  auto indexesb = pb->GetIndexes();
  for(size_t i = 0; i < axes_set[0].size(); ++i){
    assert(indexesa[axes_set[0][i]] == InverseIndex(indexesb[axes_set[1][i]]));
  }
  // Check if idx_a_ is in the outer legs
  auto iter = find(axes_set[1].begin(),axes_set[1].end(), idx_b);
  assert(iter == axes_set[1].cend());
  // Check if qn_sector_idx_a_ < number of quantum number sector
  assert(qn_sector_idx_b_ < pb->GetIndexes()[idx_b_].GetQNSctNum());
#endif
  TenCtrctInitResTen(pa_, pb_, axes_set_, pc_);
  //Then we assign the DataBlk and Allocate the memory with set 0 for pc_
  
  raw_data_ctrct_tasks_ = pc_->GetBlkSparDataTen().DataBlkGenForTenCtrct(
                              pa_->GetBlkSparDataTen(),
                              idx_a_,
                              qn_sector_idx_a_,
                              pb_->GetBlkSparDataTen(),
                              idx_b_,
                              qn_sector_idx_b_,
                              axes_set_
                          );

  SetStatus(ExecutorStatus::INITED);
}



/**
Allocate memory and perform raw data contraction calculation.
*/
template <typename TenElemT, typename QNT>
void TensorContraction1SectorExecutor<TenElemT, QNT>::Execute(void) {
  SetStatus(ExecutorStatus::EXEING);

  pc_->GetBlkSparDataTen().CtrctTwoBSDTAndAssignIn(
      pa_->GetBlkSparDataTen(),
      pb_->GetBlkSparDataTen(),
      raw_data_ctrct_tasks_
  );

  SetStatus(ExecutorStatus::FINISH);
}


/**
Function version for tensor contraction, with one of index restrict on only one sector

@tparam TenElemT The type of tensor elements.
@tparam QNT The quantum number type of the tensors.

@param pa Pointer to input tensor \f$ A \f$.
@param idx_a the idx_a-th leg is restrict.
@param qn_sector_idx_a restrict on qn_sector_idx_a-th quantum number sector.
@param pb Pointer to input tensor \f$ B \f$.
@param axes_set To-be contracted tensor axes indexes. For example, {{0, 1}, {3, 2}}.
@param pc Pointer to result tensor \f$ C \f$.
*/
template <typename TenElemT, typename QNT>
void Contract1Sector(
    const GQTensor<TenElemT, QNT> *pa,
    const size_t idx_a,
    const size_t qn_sector_idx_a,
    const GQTensor<TenElemT, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<TenElemT, QNT> *pc
) {
  TensorContraction1SectorExecutor<TenElemT, QNT> ten_ctrct_1sector_executor(
      pa,
      idx_a,
      qn_sector_idx_a,
      pb,
      axes_set,
      pc
  );
  ten_ctrct_1sector_executor.Execute();
}


template <typename QNT>
void Contract1Sector(
    const GQTensor<GQTEN_Double, QNT> *pa,
    const size_t idx_a,
    const size_t qn_sector_idx_a,
    const GQTensor<GQTEN_Complex, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<GQTEN_Complex, QNT> *pc
) {
  auto cplx_a = ToComplex(*pa);
  Contract1Sector(&cplx_a, idx_a, qn_sector_idx_a,  pb, axes_set, pc);
}


template <typename QNT>
void Contract1Sector(
    const GQTensor<GQTEN_Complex, QNT> *pa,
    const size_t idx_a,
    const size_t qn_sector_idx_a,
    const GQTensor<GQTEN_Double, QNT> *pb,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<GQTEN_Complex, QNT> *pc
) {
  auto cplx_b = ToComplex(*pb);
  Contract1Sector(pa, idx_a, qn_sector_idx_a, &cplx_b, axes_set, pc);
}




template <typename TenElemT, typename QNT>
void Contract1Sector(
    const GQTensor<TenElemT, QNT> *pa,
    const GQTensor<TenElemT, QNT> *pb,
    const size_t idx_b,
    const size_t qn_sector_idx_b,
    const std::vector<std::vector<size_t>> &axes_set,
    GQTensor<TenElemT, QNT> *pc
) {
  TensorContraction1SectorExecutor<TenElemT, QNT> ten_ctrct_1sector_executor(
      pa,
      pb,
      idx_b,
      qn_sector_idx_b,
      axes_set,
      pc
  );
  ten_ctrct_1sector_executor.Execute();
}



} /* gqten */
#endif /* ifndef GQTEN_TENSOR_MANIPULATION_TEN_CTRCT_H */
