// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-7-26
*
* Description: GraceQ/tensor project. Unittests for Boost Serialization of GQTensor.
* Note: This serialization only serilizes the wraps of tensor, no raw data
*/

#include "gqten/gqtensor_all.h"      // GQTensor, Index, QN, U1QNVal, QNSectorVec
#include "gqten/utility/utils_inl.h"        // GenAllCoors

#include "gtest/gtest.h"
#include "../testing_utility.h"     // RandInt, RandUnsignedInt, TransCoors

#include <fstream>    // ifstream, ofstream
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace gqten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

struct TestGQTensor : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 = U1QN({QNCard(qn_nm, U1QNVal(0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal(1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal(2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  QNSctT qnsct0_s = QNSctT(qn0, 4);
  QNSctT qnsctp1_s = QNSctT(qnp1, 5);
  QNSctT qnsctm1_s = QNSctT(qnm1, 3);
  QNSctT qnsct0_l = QNSctT(qn0, 10);
  QNSctT qnsctp1_l = QNSctT(qnp1, 8);
  QNSctT qnsctm1_l = QNSctT(qnm1, 12);
  IndexT idx_in_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::OUT);
  IndexT idx_in_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, GQTenIndexDirType::IN);
  IndexT idx_out_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, GQTenIndexDirType::OUT);

  DGQTensor dten_default = DGQTensor();
  DGQTensor dten_scalar = DGQTensor(IndexVec<U1QN>{});
  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_1d_l = DGQTensor({idx_out_l});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_2d_l = DGQTensor({idx_in_l, idx_out_l});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  DGQTensor dten_3d_l = DGQTensor({idx_in_l, idx_out_l, idx_out_l});
  ZGQTensor zten_default = ZGQTensor();
  ZGQTensor zten_scalar = ZGQTensor(IndexVec<U1QN>{});
  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_1d_l = ZGQTensor({idx_out_l});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_2d_l = ZGQTensor({idx_in_l, idx_out_l});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
  ZGQTensor zten_3d_l = ZGQTensor({idx_in_l, idx_out_l, idx_out_l});
};

TEST_F(TestGQTensor, TestSerialization) {
  std::ofstream ofs("dten_default.gqten");
  boost::archive::binary_oarchive oa(ofs);
  dten_scalar() = 0.3;
  dten_scalar.Show();
  oa << dten_scalar;
  ofs.close();

  std::ifstream ifs("dten_default.gqten");
  boost::archive::binary_iarchive ia(ifs);

  DGQTensor dten_scalar2;
  ia >> dten_scalar2;
  dten_scalar2.Show();
  ifs.close();

  ofs.open("dten_default.gqten");
  dten_2d_s(0, 0) = 0.4;
  oa << dten_2d_s;
  ofs.close();

  ifs.open("dten_default.gqten");
  dten_scalar2 = DGQTensor();

  ia >> dten_scalar2;
  dten_scalar2.Show();
  // EXPECT_EQ(dten_scalar2, dten_2d_s);
}

//helper
IndexT RandIndex(const unsigned qn_sct_num,  //how many quantum number sectors?
                 const unsigned max_dim_in_one_qn_sct, // maximum dimension in every quantum number sector?
                 const GQTenIndexDirType dir) {
  QNSectorVec<U1QN> qnsv(qn_sct_num);
  for (size_t i = 0; i < qn_sct_num; i++) {
    auto qn = U1QN({QNCard("qn", U1QNVal(i))});
    srand((unsigned) time(NULL));
    unsigned degeneracy = rand() % max_dim_in_one_qn_sct + 1;
    qnsv[i] = QNSector(qn, degeneracy);
  }
  return Index(qnsv, dir);
}

TEST_F(TestGQTensor, TestSerializationRandomTensor) {
  std::ofstream ofs("dten_default.gqten");
  boost::archive::binary_oarchive oa(ofs);

  auto index1_in = RandIndex(5, 4, gqten::IN);
  auto index2_in = RandIndex(4, 6, gqten::IN);
  auto index1_out = RandIndex(3, 3, gqten::OUT);
  auto index2_out = RandIndex(2, 5, gqten::OUT);

  DGQTensor t1({index2_out, index1_in, index2_in, index1_out});
  t1.Random(qn0);
  t1.ConciseShow();
  oa << t1;
  ofs.close();

  std::ifstream ifs("dten_default.gqten");
  boost::archive::binary_iarchive ia(ifs);

  DGQTensor t2;
  ia >> t2;
  ifs.close();
  // EXPECT_EQ(t1, t2);
}