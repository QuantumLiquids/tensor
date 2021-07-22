// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author:  Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn?
* Creation Date: 2021-07-22
*
* Description: GraceQ/tensor project. Unittests for tensor fuse index.
*/

#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_fuse_index.h"

#include "gtest/gtest.h"
#include "gqten/utility/timer.h"


using namespace gqten;
using U1QN = QN<U1QNVal>;
using QNSctT = QNSector<U1QN>;
using IndexT = Index<U1QN>;
using DGQTensor = GQTensor<GQTEN_Double, U1QN>;


std::string qn_nm = "qn";
U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});

QNSctT qnsct0_1 = QNSctT(qn0, 1);
QNSctT qnsct0_2 = QNSctT(qn0, 2);
QNSctT qnsct0_4 = QNSctT(qn0, 4);
QNSctT qnsctp1_1 = QNSctT(qnp1, 1);
QNSctT qnsctp1_2 = QNSctT(qnp1, 2);
QNSctT qnsctp1_4 = QNSctT(qnp1, 4);
QNSctT qnsctm1_1 = QNSctT(qnm1, 1);
QNSctT qnsctp2_1 = QNSctT(qnp2, 1);

IndexT idx_in0 = IndexT({qnsct0_1}, IN);
IndexT idx_out0 = IndexT({qnsct0_1}, OUT);
IndexT idx_out0_2 = IndexT({qnsct0_2}, OUT);
IndexT idx_out0_4 = IndexT({qnsct0_4}, OUT);

IndexT idx_in1 = IndexT({qnsct0_2, qnsctp1_1},IN);
IndexT idx_in1plus1 = IndexT({qnsct0_4, qnsctp1_4, qnsctp2_1},IN);
IndexT idx_out1 = IndexT({qnsctp1_1, qnsct0_1, qnsctm1_1},OUT);


template <typename TenT>
void RunTestTenFuseIndexCase(
    TenT &a,
    const size_t idx1,
    const size_t idx2,
    TenT &correct_res
) {
  a.FuseIndex(idx1,idx2);
  a.Show();
  correct_res.Show();
  EXPECT_TRUE(a == correct_res);
}

TEST(TestExpand, TestCase) {
  DGQTensor ten0 = DGQTensor({idx_out0_2, idx_out0_2, idx_in0});
  DGQTensor ten1 = DGQTensor({idx_out0_4, idx_in0});
  ten0({0,0,0}) = 0.5;
  ten0({0,1,0}) = 0.7;
  ten0({1,0,0}) = 0.2;
  ten0({1,1,0}) = 1.2;

  ten1({0,0}) = 0.5;
  ten1({1,0}) = 0.7;
  ten1({2,0}) = 0.2;
  ten1({3,0}) = 1.2;
  RunTestTenFuseIndexCase(ten0,0,1,ten1);

  DGQTensor ten2 = DGQTensor({idx_in1, idx_in1});
  ten2({1,2}) = 0.5; 
  ten2({2,0}) = 4.5;
  ten2({2,1}) = 2.3;
  DGQTensor ten3 = DGQTensor({idx_in1plus1});
  ten3({5}) = 0.5;
  ten3({6}) = 4.5;
  ten3({7})  = 2.3;
  RunTestTenFuseIndexCase(ten2,0,1,ten3);

}

