// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-26 19:00
*
* Description: GraceQ/tensor project. Implementation details for symmetry-blocked
* sparse tensor class.
*/

/**
@file gqtensor_impl.h
@brief  Implementation details for symmetry-blocked sparse tensor class.
*/
#ifndef GQTEN_GQTENSOR_GQTENSOR_IMPL_H
#define GQTEN_GQTENSOR_GQTENSOR_IMPL_H


#include "gqten/gqtensor/gqtensor.h"                                // GQTensor
#include "gqten/gqtensor/index.h"                                   // IndexVec, GetQNSctNumOfIdxs, CalcDiv
#include "gqten/gqtensor/blk_spar_data_ten/blk_spar_data_ten.h"     // BlockSparseDataTensor
#include "gqten/utility/utils_inl.h"                                // GenAllCoors, Rand, Reorder, CalcScalarNorm, CalcConj

#include <iostream>     // cout, endl, istream, ostream
#include <iterator>     // next
#include <algorithm>    // is_sorted

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/request.hpp>

#ifdef Release
  #define NDEBUG
#endif
#include <assert.h>     // assert


namespace gqten {


/**
Create an empty GQTensor using indexes.

@param indexes Vector of Index of the tensor.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT>::GQTensor(
    const IndexVec<QNT> &indexes
) : indexes_(indexes) {
  rank_ = indexes_.size();
  shape_ = CalcShape_();
  size_ = CalcSize_();
  if (!IsDefault()) {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
  }
}


/**
Create an empty GQTensor by moving indexes.

@param indexes Vector of Index of the tensor.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT>::GQTensor(IndexVec<QNT> &&indexes) : indexes_(indexes) {
  rank_ = indexes_.size();
  shape_ = CalcShape_();
  size_ = CalcSize_();
  if (!IsDefault()) {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
  }
}


/**
Copy a GQTensor.

@param gqten Another GQTensor.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT>::GQTensor(const GQTensor &gqten) :
    rank_(gqten.rank_),
    shape_(gqten.shape_),
    size_(gqten.size_),
    indexes_(gqten.indexes_) {
  if (gqten.IsDefault()) {
    // Do nothing
  } else {
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(
                              *gqten.pblk_spar_data_ten_
                          );
    pblk_spar_data_ten_->pgqten_indexes = &indexes_;
  }
}


/**
Assign a GQTensor.

@param rhs Another GQTensor.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> &GQTensor<ElemT, QNT>::operator=(const GQTensor &rhs) {
  rank_ = rhs.rank_;
  shape_ = rhs.shape_;
  size_ = rhs.size_;
  indexes_ = rhs.indexes_;
  if (rhs.IsDefault()) {
    // Do nothing
  } else {
    delete pblk_spar_data_ten_;
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(
                              *rhs.pblk_spar_data_ten_
                          );
    pblk_spar_data_ten_->pgqten_indexes = &indexes_;
  }
  return *this;
}


/**
Move a GQTensor.

@param gqten Another GQTensor to-be moved.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT>::GQTensor(GQTensor &&gqten) noexcept :
    rank_(gqten.rank_),
    shape_(gqten.shape_),
    size_(gqten.size_) {
  if (gqten.IsDefault()) {
    // Do nothing
  } else {
    indexes_ = std::move(gqten.indexes_);
    pblk_spar_data_ten_ = gqten.pblk_spar_data_ten_;
    gqten.pblk_spar_data_ten_ = nullptr;
    pblk_spar_data_ten_->pgqten_indexes = &indexes_;
  }
}


/**
Move and assign a GQTensor.

@param rhs Another GQTensor to-be moved.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> &GQTensor<ElemT, QNT>::operator=(GQTensor &&rhs) noexcept {
  rank_ = rhs.rank_;
  shape_ = rhs.shape_;
  size_ = rhs.size_;
  if (rhs.IsDefault()) {
    // Do nothing
  } else {
    indexes_ = std::move(rhs.indexes_);
    delete pblk_spar_data_ten_;
    pblk_spar_data_ten_ = rhs.pblk_spar_data_ten_;
    rhs.pblk_spar_data_ten_ = nullptr;
    pblk_spar_data_ten_->pgqten_indexes = &indexes_;
  }
  return *this;
}


template <typename ElemT, typename QNT>
struct GQTensor<ElemT, QNT>:: GQTensorElementAccessDeref {
  const GQTensor &const_this_ten;
  std::vector<size_t> coors;

  GQTensorElementAccessDeref(
      const GQTensor &gqten,
      const std::vector<size_t> &coors
  ) : const_this_ten(gqten), coors(coors) {
    assert(const_this_ten.Rank() == coors.size());
  }

  operator ElemT() const {
    return const_this_ten.GetElem(coors);
  }

  void operator=(const ElemT elem) {
    const_cast<GQTensor &>(const_this_ten).SetElem(coors, elem);
  }

  bool operator==(const ElemT elem) const {
    return const_this_ten.GetElem(coors) == elem;
  }

  bool operator!=(const ElemT elem) const {
    return !(*this == elem);
  }
};


/**
Access to the tensor element using its coordinates.

@param coors The coordinates of the element.
*/
template <typename ElemT, typename QNT>
typename GQTensor<ElemT, QNT>::GQTensorElementAccessDeref
GQTensor<ElemT, QNT>::operator()(const std::vector<size_t> &coors) {
    return GQTensorElementAccessDeref(*this, coors);
}


template <typename ElemT, typename QNT>
typename GQTensor<ElemT, QNT>::GQTensorElementAccessDeref
GQTensor<ElemT, QNT>::operator()(const std::vector<size_t> &coors) const {
    return GQTensorElementAccessDeref(*this, coors);
}


/**
Access to the rank 0 (scalar) tensor element.
*/
template <typename ElemT, typename QNT>
typename GQTensor<ElemT, QNT>::GQTensorElementAccessDeref
GQTensor<ElemT, QNT>::operator()(void) {
  assert(IsScalar());
  return GQTensorElementAccessDeref(*this, {});
}


template <typename ElemT, typename QNT>
typename GQTensor<ElemT, QNT>::GQTensorElementAccessDeref
GQTensor<ElemT, QNT>::operator()(void) const {
  assert(IsScalar());
  return GQTensorElementAccessDeref(*this, {});
}


/**
Access to the tensor element of a non-scalar tensor.

@tparam OtherCoorsT The types of the second, third, etc coordinates.
@param coor0 The first coordinate.
@param other_coors The second, third, ... coordiantes. They should be
       non-negative integers.
*/
template <typename ElemT, typename QNT>
template <typename... OtherCoorsT>
typename GQTensor<ElemT, QNT>::GQTensorElementAccessDeref
GQTensor<ElemT, QNT>::operator()(
    const size_t coor0,
    const OtherCoorsT... other_coors
) {
  return GQTensorElementAccessDeref(
      *this,
      {coor0, static_cast<size_t>(other_coors)...}
  );
}


template <typename ElemT, typename QNT>
template <typename... OtherCoorsT>
typename GQTensor<ElemT, QNT>::GQTensorElementAccessDeref
GQTensor<ElemT, QNT>::operator()(
    const size_t coor0,
    const OtherCoorsT... other_coors
) const {
  return GQTensorElementAccessDeref(
      *this,
      {coor0, static_cast<size_t>(other_coors)...}
  );
}


/**
Calculate the quantum number divergence of the GQTensor.

@return The quantum number divergence.
*/
template <typename ElemT, typename QNT>
QNT GQTensor<ElemT, QNT>::Div(void) const {
  assert(!IsDefault());
  if (IsScalar()) {
    std::cout << "Tensor is a scalar. Return empty quantum number."
              << std::endl;
    return QNT();
  } else {
    auto qnblk_num = GetQNBlkNum();
    if (qnblk_num == 0) {
      std::cout << "Tensor does not have a block. Return empty quantum number."
                << std::endl;
      return QNT();
    } else {
      auto blk_idx_data_blk_map = GetBlkSparDataTen().GetBlkIdxDataBlkMap();
      auto indexes = GetIndexes();
      auto first_blk_idx_data_blk = blk_idx_data_blk_map.begin();
      auto div = CalcDiv(indexes, first_blk_idx_data_blk->second.blk_coors);
      for (
          auto it = std::next(first_blk_idx_data_blk);
          it != blk_idx_data_blk_map.end();
          ++it
      ) {
        auto blki_div = CalcDiv(indexes, it->second.blk_coors);
        if (blki_div != div) {
          std::cout << "Tensor does not have a special divergence. Return empty quantum number."
                    << std::endl;
          return QNT();
        }
      }
      return div;
    }
  }
}


/**
Get the tensor element using its coordinates.

@coors The coordinates of the tensor element. An empty vector for scalar.
*/
template <typename ElemT, typename QNT>
ElemT GQTensor<ElemT, QNT>::GetElem(const std::vector<size_t> &coors) const {
  assert(!IsDefault());
  assert(coors.size() == rank_);
  auto blk_coors_data_coors = CoorsToBlkCoorsDataCoors_(coors);
  return pblk_spar_data_ten_->ElemGet(blk_coors_data_coors);
}


/**
Set the tensor element using its coordinates.

@param coors The coordinates of the tensor element. An empty vector for scalar.
@param elem The value of the tensor element.
*/
template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::SetElem(
    const std::vector<size_t> &coors,
    const ElemT elem
) {
  assert(!IsDefault());
  assert(coors.size() == rank_);
  auto blk_coors_data_coors = CoorsToBlkCoorsDataCoors_(coors);
  pblk_spar_data_ten_->ElemSet(blk_coors_data_coors, elem);
}


/**
Equivalence check.

@param rhs The GQTensor at the right hand side.

@return Equivalence check result.
*/
template <typename ElemT, typename QNT>
bool GQTensor<ElemT, QNT>::operator==(const GQTensor &rhs) const {
  // Default check
  if (IsDefault() || rhs.IsDefault()) {
    if (IsDefault() && rhs.IsDefault()) {
      return true;
    } else {
      return false;
    }
  }
  // Indexes check
  if (indexes_ != rhs.indexes_) { return false; }
  // Block sparse data tensor check
  return (*pblk_spar_data_ten_ == *rhs.pblk_spar_data_ten_);
}


/**
Random set tensor elements in [0, 1] with given quantum number divergence.
Original data of this tensor will be destroyed.

@param div Target quantum number divergence.
*/
template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::Random(const QNT &div) {
  assert(!IsDefault());
  if (IsScalar()) { assert(div == QNT()); }
  pblk_spar_data_ten_->Clear();
  if(IsScalar()){
    pblk_spar_data_ten_->Random();
    return;
  }
  if(pblk_spar_data_ten_->blk_shape.size()==2){
    ShapeT shape = pblk_spar_data_ten_->blk_shape;
    std::vector<CoorsT> blk_coors_s;
    blk_coors_s.reserve(shape[0]);
      for(size_t i=0; i<shape[0];i++){
        for(size_t j=0; j<shape[1];j++){
          if (CalcDiv(indexes_, {i,j}) == div) {
            blk_coors_s.push_back({i,j});
          }
        }
      }
    pblk_spar_data_ten_->DataBlksInsert(blk_coors_s, false,false);     // NO allocate memory on this stage.
  }else{
    for (auto &blk_coors : GenAllCoors(pblk_spar_data_ten_->blk_shape)) {
      if (CalcDiv(indexes_, blk_coors) == div) {
        pblk_spar_data_ten_->DataBlkInsert(blk_coors, false);     // NO allocate memory on this stage.
      }
    }
  }
  pblk_spar_data_ten_->Random();
}


/**
Transpose the tensor using a new indexes order.

@param transed_idxes_order Transposed order of indexes.
*/
template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::Transpose(
    const std::vector<size_t> &transed_idxes_order
) {
  assert(!IsDefault());

  if (IsScalar()) { return; }

  assert(transed_idxes_order.size() == rank_);
  // Give a shorted order, do nothing
  if (std::is_sorted(transed_idxes_order.begin(), transed_idxes_order.end())) {
    return;
  }
  Reorder(shape_, transed_idxes_order);
  Reorder(indexes_, transed_idxes_order);
  pblk_spar_data_ten_->Transpose(transed_idxes_order);
}


/**
Normalize the tensor and return its norm.

@return The norm before the normalization.
*/
template <typename ElemT, typename QNT>
GQTEN_Double GQTensor<ElemT, QNT>::Normalize(void) {
  assert(!IsDefault());
  GQTEN_Double norm = pblk_spar_data_ten_->Normalize();
  return norm;
}


/**
Switch the direction of the indexes, complex conjugate of the elements.
*/
template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::Dag(void) {
  assert(!IsDefault());
  for (auto &index : indexes_) { index.Inverse(); }
  pblk_spar_data_ten_->Conj();

}


/**
Calculate \f$ -1 * T \f$.

@return \f$ -T \f$.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> GQTensor<ElemT, QNT>::operator-(void) const {
  assert(!IsDefault());
  GQTensor<ElemT, QNT> res(*this);
  res *= -1.0;
  return res;
}


/**
Add this GQTensor \f$ A \f$ and another GQTensor \f$ B \f$.

@param rhs Another GQTensor \f$ B \f$.

@return \f$ A + B \f$.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> GQTensor<ElemT, QNT>::operator+(
    const GQTensor &rhs
) const {
  assert(!(IsDefault() || rhs.IsDefault()));
  assert(indexes_ == rhs.indexes_);
  GQTensor<ElemT, QNT> res(indexes_);
  res.pblk_spar_data_ten_->AddTwoBSDTAndAssignIn(
      *pblk_spar_data_ten_,
      *rhs.pblk_spar_data_ten_
      );
  return res;
}


/**
Add and assign another GQTensor \f$ B \f$ to this tensor.

@param rhs Another GQTensor \f$ B \f$.

@return \f$ A = A + B \f$.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> &GQTensor<ElemT, QNT>::operator+=(const GQTensor &rhs) {
  assert(!(IsDefault() || rhs.IsDefault()));
  assert(indexes_ == rhs.indexes_);
  pblk_spar_data_ten_->AddAndAssignIn(*rhs.pblk_spar_data_ten_);
  return *this;
}


/**
Multiply a GQTensor by a scalar (real/complex number).

@param s A scalar.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> GQTensor<ElemT, QNT>::operator*(const ElemT s) const {
  assert(!IsDefault());
  GQTensor<ElemT, QNT> res(*this);
  res.pblk_spar_data_ten_->MultiplyByScalar(s);
  return res;
}


/**
Multiply a GQTensor by a scalar (real/complex number) and assign back.

@param s A scalar.
*/
template <typename ElemT, typename QNT>
GQTensor<ElemT, QNT> &GQTensor<ElemT, QNT>::operator*=(const ElemT s) {
  assert(!IsDefault());
  pblk_spar_data_ten_->MultiplyByScalar(s);
  return *this;
}


template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::StreamRead(std::istream &is) {
  assert(IsDefault());    // Only default tensor can read data
  is >> rank_;
  indexes_ = IndexVec<QNT>(rank_);
  for (auto &index : indexes_) { is >> index; }
  shape_ = CalcShape_();
  size_ = CalcSize_();
  pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
  is >> (*pblk_spar_data_ten_);
}


/**
Boost serialization
*/
template <typename ElemT, typename QNT>
template <class Archive>
void GQTensor<ElemT, QNT>::load(Archive &ar, const unsigned int version){
  assert(IsDefault());
  ar & rank_;
  ar & indexes_;
  ar & shape_;
  ar & size_;
  if(size_>0){// not default
    pblk_spar_data_ten_ = new BlockSparseDataTensor<ElemT, QNT>(&indexes_);
    ar & (*pblk_spar_data_ten_);
  }
}


template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::StreamWrite(std::ostream &os) const {
  assert(!IsDefault());
  os << rank_ << std::endl;
  for (auto &index : indexes_) { os << index; }
  os << (*pblk_spar_data_ten_);
}


/**
Boost serialization
*/
template <typename ElemT, typename QNT>
template <class Archive>
void GQTensor<ElemT, QNT>::save(Archive &ar, const unsigned int version) const{
  assert(!IsDefault());
  ar & rank_;
  ar & indexes_;
  ar & shape_;
  ar & size_;
  ar & (*pblk_spar_data_ten_);
}





template <typename VecElemT>
inline void VectorPrinter(const std::vector<VecElemT> &vec) {
  auto size = vec.size();
  for (size_t i = 0; i < size; ++i) {
    std::cout << vec[i];
    if (i != size - 1) {
      std::cout << ", ";
    }
  }
}


template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level) << "GQTensor:" << std::endl;
  std::cout << IndentPrinter(indent_level + 1) << "Shape: (";
  VectorPrinter(shape_);
  std::cout << ")" << std::endl;
  std::cout << IndentPrinter(indent_level + 1) << "Indices:" << std::endl;
  for (auto &index : indexes_) {
    index.Show(indent_level + 2);
  }
  if (!IsDefault()) {
    std::cout << IndentPrinter(indent_level + 1) << "Divergence:" << std::endl;
    Div().Show(indent_level + 2);
    std::cout << IndentPrinter(indent_level + 1) << "Nonzero elements:" << std::endl;
    for (auto &coors : GenAllCoors(shape_)) {
      auto elem = GetElem(coors);
      if (elem != ElemT(0.0)) {
        std::cout << IndentPrinter(indent_level + 2) << "[";
        VectorPrinter(coors);
        std::cout << "]: " << elem << std::endl;
      }
    }
  }
}

template <typename QNT>
std::string ElemenTypeOfTensor(const GQTensor<GQTEN_Double,QNT> & t){
    return "GQTEN_Double";
}

template <typename QNT>
std::string ElemenTypeOfTensor(const GQTensor<GQTEN_Complex,QNT> & t){
    return "GQTEN_Complex";
}


template <typename ElemT, typename QNT>
void GQTensor<ElemT, QNT>::ConciseShow(const size_t indent_level) const{
    using std::cout;
    using std::endl;
    cout  << IndentPrinter(indent_level) << "GQTensor Concise Info: " <<"\n";
    cout  << IndentPrinter(indent_level+1) << "tensor shape:" << "\t[";
    VectorPrinter(shape_);
    cout  << "]\n";
    cout  << IndentPrinter(indent_level+1) << "tensor elementary type:\t" 
          << ElemenTypeOfTensor(*this) << "\n";
    cout  << IndentPrinter(indent_level+1) 
          << "tensor qumtum number block number:\t" 
          << GetQNBlkNum() << "\n";
    cout  << IndentPrinter(indent_level+1)  
          << "tensor size(product of shape):\t" << size_ <<"\n";
    if(IsDefault()){
      cout << IndentPrinter(indent_level+1)  << "default tensor" <<endl;
      return;
    }
    unsigned data_size = pblk_spar_data_ten_->GetActualRawDataSize();
    cout  << IndentPrinter(indent_level+1) 
          << "actual data size:\t" << data_size << "\n";
    cout  << IndentPrinter(indent_level+1) 
          << "tensor sparsity:\t" << double(data_size) / double(size_) << endl;
}

/**
Calculate shape from tensor rank.
*/
template <typename ElemT, typename QNT>
inline ShapeT GQTensor<ElemT, QNT>::CalcShape_(void) const {
  ShapeT shape(rank_);
  for (size_t i = 0; i < rank_; ++i) {
    shape[i] = indexes_[i].dim();
  }
  return shape;
}


/**
Calculate size from tensor shape.
*/
template <typename ElemT, typename QNT>
inline size_t GQTensor<ElemT, QNT>::CalcSize_(void) const {
  size_t size = 1;
  for (auto dim : shape_) { size *= dim; }
  return size;
}


/**
Convert tensor element coordinates to data block coordinates and in-block data
coordinates.

@param coors Tensor element coordinates.
*/
template <typename ElemT, typename QNT>
inline
std::pair<CoorsT, CoorsT> GQTensor<ElemT, QNT>::CoorsToBlkCoorsDataCoors_(
    const CoorsT &coors
) const {
  assert(coors.size() == rank_);
  CoorsT blk_coors {};
  CoorsT data_coors {};
  for (size_t i = 0; i < coors.size(); ++i) {
    auto blk_coor_data_coor = indexes_[i].CoorToBlkCoorDataCoor(coors[i]);
    blk_coors.push_back(blk_coor_data_coor.first);
    data_coors.push_back(blk_coor_data_coor.second);
  }
  return make_pair(blk_coors, data_coors);
}


const int kMPIDataTagMultiplyFactor=11;

template <typename ElemT, typename QNT>
inline void send_gqten(boost::mpi::communicator world,
                int dest, int tag,
                const GQTensor<ElemT, QNT>& gqten){
  world.send(dest,tag,gqten.IsDefault());
  if(gqten.IsDefault()){
    return;
  }
  boost::mpi::request reqs[2];
#ifdef GQTEN_MPI_TIMING_MODE
  Timer send_gqten_wrap_timer("send_gqten_wrap");
#endif
  reqs[0] = world.isend(dest, tag, gqten);
#ifdef GQTEN_MPI_TIMING_MODE
  send_gqten_wrap_timer.PrintElapsed();

  Timer send_gqten_data_timer("send_gqten_data");
#endif
  int tag_data = tag*kMPIDataTagMultiplyFactor+1;
  const BlockSparseDataTensor<ElemT, QNT>& bsdt=gqten.GetBlkSparDataTen();
  const ElemT* data_pointer = bsdt.GetActualRawDataPtr();
  int data_size = bsdt.GetActualRawDataSize();
  ElemT zero = ElemT(0.0);
  if( gqten.IsScalar() && data_size==0 ){
    data_pointer = &zero;
    data_size = 1;
  }
  reqs[1] = world.isend(dest, tag_data, data_pointer, data_size);
#ifdef GQTEN_MPI_TIMING_MODE
  send_gqten_data_timer.PrintElapsed();
#endif
  boost::mpi::wait_all(reqs,reqs+2);
}


template <typename ElemT, typename QNT>
inline boost::mpi::status recv_gqten(boost::mpi::communicator world,
                int source, int tag,
                GQTensor<ElemT, QNT>& gqten){
  assert(gqten.IsDefault());
  bool is_default;
  boost::mpi::status recv_ten_status=world.recv(source,tag,is_default);
  if(is_default){
    return recv_ten_status;
  }
  if(source == boost::mpi::any_source){
    source = recv_ten_status.source();
    tag = recv_ten_status.tag();
  }
  // boost::mpi::request reqs[2];
#ifdef GQTEN_MPI_TIMING_MODE
  Timer recv_gqten_wrap_timer("recv_gqten_wrap");
#endif
  world.recv(source, tag, gqten);
#ifdef GQTEN_MPI_TIMING_MODE
  recv_gqten_wrap_timer.PrintElapsed();

  Timer recv_gqten_data_timer("recv_gqten_data");
#endif
  int tag_data = tag*kMPIDataTagMultiplyFactor+1;
  BlockSparseDataTensor<ElemT, QNT>& bsdt=gqten.GetBlkSparDataTen();
  ElemT* data_pointer = bsdt.pactual_raw_data_;
  int data_size = bsdt.GetActualRawDataSize();
  auto s = world.recv(source, tag_data, data_pointer, data_size);
#ifdef GQTEN_MPI_TIMING_MODE
  recv_gqten_data_timer.PrintElapsed();
#endif
  
  // boost::mpi::wait_all(reqs+1,reqs+2);
  return s;
}


template <typename ElemT, typename QNT>
inline void SendBroadCastGQTensor(
  boost::mpi::communicator world,
  const GQTensor<ElemT, QNT>& gqten,
  const int root
){
  assert(world.rank() == root );
  bool is_default = gqten.IsDefault();
  boost::mpi::broadcast(world, is_default, root ); //also a signal start communication
  if(gqten.IsDefault()){
    return;
  }
  boost::mpi::broadcast(world, const_cast<GQTensor<ElemT, QNT>&>(gqten), root);

  const BlockSparseDataTensor<ElemT, QNT>& bsdt=gqten.GetBlkSparDataTen();
  const ElemT* raw_data_pointer = bsdt.GetActualRawDataPtr();
  int raw_data_size = bsdt.GetActualRawDataSize();
  ElemT zero = ElemT(0.0);
  if( gqten.IsScalar() && raw_data_size==0 ){
    raw_data_pointer = &zero;
    raw_data_size = 1;
  }
  //below broadcast need safely remove const
  boost::mpi::broadcast(world, const_cast<ElemT*>(raw_data_pointer), raw_data_size, root);
}

template <typename ElemT, typename QNT>
inline void RecvBroadCastGQTensor(
  boost::mpi::communicator world,
  GQTensor<ElemT, QNT>& gqten,
  const int root
){
  assert(world.rank() != root );
  assert(gqten.IsDefault());
  bool is_default;
  boost::mpi::broadcast(world, is_default, root);
  if(is_default){
    return;
  }
  boost::mpi::broadcast(world, gqten,root);

  BlockSparseDataTensor<ElemT, QNT>& bsdt=gqten.GetBlkSparDataTen();
  ElemT* raw_data_pointer = bsdt.pactual_raw_data_;
  int raw_data_size = bsdt.GetActualRawDataSize();
  boost::mpi::broadcast(world, raw_data_pointer, raw_data_size, root);
}

} /* gqten */
#endif /* ifndef GQTEN_GQTENSOR_GQTENSOR_IMPL_H */
