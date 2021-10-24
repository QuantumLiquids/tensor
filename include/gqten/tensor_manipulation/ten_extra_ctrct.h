// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-10-23.
*
* Description: tensor extra contraction function
*/

/**
@file ten_extra_ctrct.h
@brief tensor extra contraction function.
       This extra contraction is defined by the following:
       - the contracted indexes must be continuous. More concentrate, if it was defined by the usually language
            Contract(A, B, axes1, axes2), the axes1 must be ascendant numbers with interval 1, and so as axes2.
            We allow the period boundary of axes, for example: if A.rank() == 5, axes1 can be {3,4,0}
       - thus the transposes in contraction can be realized in matrix transpose
*/


#ifndef GQTEN_TENSOR_MANIPULATION_TEN_EXTRA_CTRCT_H
#define GQTEN_TENSOR_MANIPULATION_TEN_EXTRA_CTRCT_H

#include "gqten/framework/bases/executor.h"
#include "gqten/gqtensor_all.h"
#include <set>

namespace gqten{

/**
 *
 * @tparam TenElemT
 * @tparam QNT
 * @tparam a_ctrct_tail
 * @tparam b_ctrct_head
 *
 *  develop note: if a_ctrct_tail && b_ctrct_head, then no transpose need when matrix product.
 *  else transpose the corresponding the tensor(s) when matrix product
 *  this two parameters don't change the result of contraction,
 *  but support some hints to reduce operators and increase the performance.
 */
template <typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
class TensorExtraContractionExecutor: public Executor {
 public:
  TensorExtraContractionExecutor(
      const GQTensor<TenElemT, QNT>* ,
      const GQTensor<TenElemT, QNT>* ,
      const size_t a_ctrct_axes_start,
      const size_t b_ctrct_axes_start,
      const size_t ctrct_axes_size,
      GQTensor<TenElemT, QNT>*
      );
  void Execute(void) override;

 private:
  void GenerateDataBlk_();
  void TransposePrepare_();

  void ExecutePost_();//clear transpose data;

  const GQTensor<TenElemT, QNT> *pa_;
  const GQTensor<TenElemT, QNT> *pb_;
  GQTensor<TenElemT, QNT> *pc_;
  ushort a_ctrct_axes_start_; //ctrct_axes include this one
  ushort b_ctrct_axes_start_;
  ushort a_ctrct_axes_end_; //ctrct_axes do not include this one
  ushort b_ctrct_axes_end_; //ctrct_axes do not include this one
  ushort ctrct_axes_size_;

  ushort a_trans_critical_axe_;
  //if a_trans_critical_axe_ > 0, transpose happen between (a_trans_ctritical_axe_ - 1, a_trans_ctritical_axe_),
  //else a_trans_critical_axe_ == 0, no need to transpose. (this design can be reconsidered)
  ushort b_trans_critical_axe_;
  //if b_trans_critical_axe_ == 0, no need to transpose

  TenElemT* a_trans_data_ = nullptr;
  //TODO: more template parameter to determine if the original data is need to save,
  // so that we can save some memory.
  TenElemT* b_trans_data_ = nullptr;
  std::vector<RawDataCtrctTask> raw_data_ctrct_tasks_;
  //and maybe more ...
};

/**
 * @example
 *      For the general contraction: Contract(A, B, {1,2,3},{5,6,0});
 *      a_ctrct_axes_start = 1,
 *      b_ctrct_axes_start = 5,
 *      ctrct_axes_size = 3;
 *
 *      for a_ctrct_tail == true case
 *          a_trans_critical_axe_ = 4 if A.Rank()>4 or a_trans_critical_axe_ = 0 if A.Rank()==4;
 *
 *
 * @tparam TenElemT
 * @tparam QNT
 * @tparam a_ctrct_tail
 * @tparam b_ctrct_head
 * @param pa
 * @param pb
 * @param a_ctrct_axes_start
 * @param b_ctrct_axes_start
 * @param ctrct_axes_size
 * @param pc
 *
 *  question: meaning for set status?
 */
template <typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::TensorExtraContractionExecutor(
    const GQTensor<TenElemT, QNT>* pa,
    const GQTensor<TenElemT, QNT>* pb,
    const size_t a_ctrct_axes_start,
    const size_t b_ctrct_axes_start,
    const size_t ctrct_axes_size,
    GQTensor<TenElemT, QNT>* pc
) : pa_(pa), pb_(pb), pc_(pc),
a_ctrct_axes_start_(a_ctrct_axes_start),
b_ctrct_axes_start_(b_ctrct_axes_start),
a_ctrct_axes_end_((a_ctrct_axes_start + ctrct_axes_size)%(pa->Rank())),
b_ctrct_axes_end_((b_ctrct_axes_start + ctrct_axes_size)%(pb->Rank())),
ctrct_axes_size_(ctrct_axes_size),
a_trans_critical_axe_( a_ctrct_tail ? a_ctrct_axes_end_ : a_ctrct_axes_start),
b_trans_critical_axe_( b_ctrct_head ? b_ctrct_axes_start : b_ctrct_axes_end_) {}


template <typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::GenerateDataBlk_() {
  using std::vector;
  const ushort a_rank(pa_->Rank()), b_rank(pb_->Rank());
  vector<vector<size_t>> saved_axes_set(2), ctrct_axes_set(2);
  ctrct_axes_set[0].reserve(ctrct_axes_size_);
  ctrct_axes_set[1].reserve(ctrct_axes_size_);
  saved_axes_set[0].reserve(pa_->Rank() - ctrct_axes_size_);
  saved_axes_set[1].reserve(pb_->Rank() - ctrct_axes_size_);
  for(ushort i = 0; i < ctrct_axes_size_; i ++ ) {
    ctrct_axes_set[0].push_back((a_ctrct_axes_start_+i)%a_rank);
    ctrct_axes_set[1].push_back((b_ctrct_axes_start_+i)%b_rank);
  }
  const ushort save_axes_size_a = a_rank - ctrct_axes_size_;
  const ushort save_axes_size_b = b_rank - ctrct_axes_size_;
  for(ushort i = 0; i < save_axes_size_a; i ++) {
    saved_axes_set[0].push_back( (a_ctrct_axes_end_+i)%a_rank);
  }
  for(ushort i = 0; i < save_axes_size_b; i ++) {
    saved_axes_set[1].push_back( (b_ctrct_axes_end_+i)%b_rank);
  }

  TenCtrctInitResTen(pa_, pb_, ctrct_axes_set, pc_);
  raw_data_ctrct_tasks_ = pc_->GetBlkSparDataTen().DataBlkGenForTenCtrct(
      pa_->GetBlkSparDataTen(),
      pb_->GetBlkSparDataTen(),
      ctrct_axes_set,
      saved_axes_set
  );
}

template <typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::TransposePrepare_() {
  using std::set;
  if(a_trans_critical_axe_ > 0 ){
    const auto& a_bsdt = pa_->GetBlkSparDataTen();
    const size_t a_raw_data_size = a_bsdt.GetActualRawDataSize();
    a_trans_data_ = (TenElemT*) malloc( a_raw_data_size *sizeof(TenElemT) );
    set<size_t> selected_data_blk_idxs;
    for(auto& task : raw_data_ctrct_tasks_){
      selected_data_blk_idxs.insert(task.a_blk_idx);
    }
    a_bsdt.OutOfPlaceMatrixTransposeForSelectedDataBlk(
        selected_data_blk_idxs,
        a_trans_critical_axe_,
        a_trans_data_
        );
  }
  if(b_trans_critical_axe_ > 0) {
    const auto& b_bsdt = pb_->GetBlkSparDataTen();
    const size_t b_raw_data_size = b_bsdt.GetActualRawDataSize();
    b_trans_data_ = (TenElemT*) malloc( b_raw_data_size *sizeof(TenElemT) );
    set<size_t> selected_data_blk_idxs;
    for(auto& task : raw_data_ctrct_tasks_){
      selected_data_blk_idxs.insert(task.b_blk_idx);
    }
    b_bsdt.OutOfPlaceMatrixTransposeForSelectedDataBlk(
        selected_data_blk_idxs,
        b_trans_critical_axe_,
        b_trans_data_
    );
  }
}

template <typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::ExecutePost_() {
  free(a_trans_data_);
  free(b_trans_data_);
}

template <typename TenElemT, typename QNT, bool a_ctrct_tail, bool b_ctrct_head>
void TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>::Execute() {
  GenerateDataBlk_();
  TransposePrepare_();

  const TenElemT* a_raw_data;
  const TenElemT* b_raw_data;
  if( a_trans_critical_axe_ > 0){
    a_raw_data = a_trans_data_;
  } else{
    a_raw_data = pa_->GetBlkSparDataTen().GetActualRawDataPtr();
  }

  if( b_trans_critical_axe_ > 0){
    b_raw_data = b_trans_data_;
  } else{
    b_raw_data = pb_->GetBlkSparDataTen().GetActualRawDataPtr();
  }
  auto& bsdt_c = pc_->GetBlkSparDataTen();
  bsdt_c.template CtrctAccordingTask<a_ctrct_tail, b_ctrct_head>(
      a_raw_data,
      b_raw_data,
      raw_data_ctrct_tasks_
      );
  ExecutePost_();
}

/**
 *
 * @tparam TenElemT
 * @tparam QNT
 * @tparam a_ctrct_use_tail
 * @tparam b_ctrct_use_head
 * @param pa
 * @param pb
 * @param a_ctrct_axes_start
 * @param a_ctrct_axes_size
 * @param b_ctrct_axes_start
 * @param b_ctrct_axes_size
 * @param pc
 */
template <typename TenElemT, typename QNT, bool a_ctrct_tail = true, bool b_ctrct_head = true>
void Contract(
    const GQTensor<TenElemT, QNT>& pa, //use ref to make sure it is not a null pointer
    const GQTensor<TenElemT, QNT>& pb, //TODO: unify the style of code
    const size_t a_ctrct_axes_start,
    const size_t b_ctrct_axes_start,
    const size_t ctrct_axes_size,
    GQTensor<TenElemT, QNT>& pc
) {
  auto extra_contraction_executor =  TensorExtraContractionExecutor<TenElemT, QNT, a_ctrct_tail, b_ctrct_head>(
    &pa, &pb, a_ctrct_axes_start, b_ctrct_axes_start, ctrct_axes_size, &pc
      );
  extra_contraction_executor.Execute();
}


}//gqten



#endif //GQTEN_TENSOR_MANIPULATION_TEN_EXTRA_CTRCT_H
