// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-8-20
*
* Description: GraceQ/tensor project. MPI SVD for a symmetric GQTensor.
*/

/**
@file mpi_svd.h
@brief MPI SVD for a symmetric GQTensor.
*/

#ifndef GQTEN_MPI_TENSOR_MANIPULATION_TEN_DECOMP_MPI_SVD_H
#define GQTEN_MPI_TENSOR_MANIPULATION_TEN_DECOMP_MPI_SVD_H

#include "gqten/framework/bases/executor.h"                           // Executor
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_decomp/ten_decomp_basic.h"    // GenIdxTenDecompDataBlkMats, IdxDataBlkMatMap
#include "gqten/tensor_manipulation/ten_decomp/ten_svd.h"             // TensorSVDExecutor
#include "boost/mpi.hpp"                                              // communicator

namespace gqten {
/**
MPI Tensor SVD executor used in master processor.

@tparam TenElemT Element type of tensors.
@tparam QNT Quantum number type of tensors.
*/
template<typename TenElemT, typename QNT>
class MPITensorSVDExecutor : public TensorSVDExecutor<TenElemT, QNT> {
 public:
  MPITensorSVDExecutor(
      const GQTensor<TenElemT, QNT> *,
      const size_t,
      const QNT &,
      const GQTEN_Double, const size_t, const size_t,
      GQTensor<TenElemT, QNT> *,
      GQTensor<GQTEN_Double, QNT> *,
      GQTensor<TenElemT, QNT> *,
      GQTEN_Double *, size_t *,
      boost::mpi::communicator &
  );

  ~MPITensorSVDExecutor(void) = default;

  void Execute(void) override;

 private:
  boost::mpi::communicator &world_;
};

template<typename TenElemT, typename QNT>
MPITensorSVDExecutor<TenElemT, QNT>::MPITensorSVDExecutor(
    const GQTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    const GQTEN_Double trunc_err, const size_t Dmin, const size_t Dmax,
    GQTensor<TenElemT, QNT> *pu,
    GQTensor<GQTEN_Double, QNT> *ps,
    GQTensor<TenElemT, QNT> *pvt,
    GQTEN_Double *pactual_trunc_err, size_t *pD,
    boost::mpi::communicator &world
): TensorSVDExecutor<TenElemT, QNT>(pt, ldims, lqndiv, trunc_err, Dmin, Dmax, pu, ps, pvt,
                                    pactual_trunc_err, pD), world_(world) {}

/**
MPI Execute tensor SVD calculation.
*/
template<typename TenElemT, typename QNT>
void MPITensorSVDExecutor<TenElemT, QNT>::Execute(void) {
  this->SetStatus(ExecutorStatus::EXEING);

  auto idx_raw_data_svd_res = this->pt_->GetBlkSparDataTen().DataBlkDecompSVDMaster(
      this->idx_ten_decomp_data_blk_mat_map_,
      world_
  );
  auto kept_sv_info = this->CalcTruncedSVInfo_(idx_raw_data_svd_res);
  this->ConstructSVDResTens_(kept_sv_info, idx_raw_data_svd_res);
  DeleteDataBlkMatSvdResMap(idx_raw_data_svd_res);

  this->SetStatus(ExecutorStatus::FINISH);
}

/**
Function version for tensor SVD.

@tparam TenElemT The element type of the tensors.
@tparam QNT The quantum number type of the tensors.

@param pt A pointer to to-be SVD decomposed tensor \f$ T \f$. The rank of \f$ T
       \f$ should be larger then 1.
@param ldims Number of indexes on the left hand side of the decomposition.
@param lqndiv Quantum number divergence of the result \f$ U \f$ tensor.
@param trunc_err The target truncation error.
@param Dmin The target minimal kept dimensions for the truncated SVD decomposition.
@param Dmax The target maximal kept dimensions for the truncated SVD decomposition.
@param pu A pointer to result \f$ U \f$ tensor.
@param ps A pointer to result \f$ S \f$ tensor.
@param pvt A pointer to result \f$ V^{\dagger} \f$ tensor.
@param pactual_trunc_err A pointer to actual truncation error after the truncation.
@param pD A pointer to actual kept dimensions after the truncation.
*/
template<typename TenElemT, typename QNT>
void MPISVDMaster(
    const GQTensor<TenElemT, QNT> *pt,
    const size_t ldims,
    const QNT &lqndiv,
    const GQTEN_Double trunc_err, const size_t Dmin, const size_t Dmax,
    GQTensor<TenElemT, QNT> *pu,
    GQTensor<GQTEN_Double, QNT> *ps,
    GQTensor<TenElemT, QNT> *pvt,
    GQTEN_Double *pactual_trunc_err, size_t *pD,
    boost::mpi::communicator &world
) {
  MPITensorSVDExecutor<TenElemT, QNT> mpi_ten_svd_executor(
      pt,
      ldims,
      lqndiv,
      trunc_err, Dmin, Dmax,
      pu, ps, pvt, pactual_trunc_err, pD,
      world
  );
  mpi_ten_svd_executor.Execute();
}

/**
 * 
 * @note claim the type TenElemT when call this function
 */
template<typename TenElemT>
inline void MPISVDSlave(
    boost::mpi::communicator &world
) {
  DataBlkDecompSVDSlave<TenElemT>(world);
}

}

#endif