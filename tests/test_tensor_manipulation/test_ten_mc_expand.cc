// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2021-03-25 09:01
*
* Description: GraceQ/tensor project. Unittests for tensor expansion.
*/
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_expand.h"
#include "gqten/tensor_manipulation/ten_mc_expand.h"
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
U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
U1QN qnm2 = U1QN({QNCard(qn_nm, U1QNVal(-2))});
QNSctT qnsctp1 = QNSctT(qnp1, 1);
QNSctT qnsctp2 = QNSctT(qnp1, 2);
QNSctT qnsctp3 = QNSctT(qnp1, 200);
QNSctT qnsctm1 = QNSctT(qnm1, 1);
QNSctT qnsctm2 = QNSctT(qnm1, 2);
QNSctT qnsctm3 = QNSctT(qnm1, 100);
QNSctT qnsct0 = QNSctT(qn0, 220);
QNSctT qnsct0_1 = QNSctT(qn0, 1);

QNSctT qnsct0_2 = QNSctT(qn0, 2);
QNSctT qnsctp1_2 = QNSctT(qnp1, 2);
QNSctT qnsctp1_4 = QNSctT(qnp1, 4);
QNSctT qnsctp2_2 = QNSctT(qnp2, 2);
QNSctT qnsctp2_4 = QNSctT(qnp2, 4);
QNSctT qnsctm1_2 = QNSctT(qnm1, 2);
QNSctT qnsctm1_4 = QNSctT(qnm1, 4);
QNSctT qnsctm2_2 = QNSctT(qnm2, 2);

IndexT idx_in0 = IndexT({qnsct0_1}, IN);
IndexT idx_in0_2 = IndexT({qnsct0_2}, IN);
IndexT idx_inm1 = IndexT({qnsctm1}, IN);
IndexT idx_inp1 = IndexT({qnsctp1}, IN);
IndexT idx_outm1 = IndexT({qnsctm1}, OUT);
IndexT idx_outm2 = IndexT({qnsctm2}, OUT);
IndexT idx_outp1 = IndexT({qnsctp1}, OUT);
IndexT idx_out1 = IndexT({qnsctp3, qnsct0,qnsctm3}, OUT);
IndexT idx_in1 = IndexT({qnsctp3, qnsct0,qnsctm3}, IN);
IndexT idx_in2 = IndexT({qnsctm1, qnsctp1}, IN);
IndexT idx_in4 = IndexT({qnsctm2, qnsctp2}, IN);
IndexT idx_out2 = IndexT({qnsctm1, qnsctp1}, OUT);


template <typename ElemT,typename QNT>
void RunTestTenMCExpansionCase(
    const GQTensor<ElemT, QNT> &a,
    const GQTensor<ElemT, QNT> &b,
    const std::vector<size_t> &expand_idx_nums,
    const GQTensor<ElemT, QNT> &c
) {
  using TenT = GQTensor<ElemT, QNT>;
  using BSDT = BlockSparseDataTensor<ElemT,QNT>;
  TenT res;
  ExpandMC(&a, &b, expand_idx_nums, &res);
  EXPECT_TRUE(res == c);
}


TEST(TestMCExpand, TestCase) {
  DGQTensor ten0 = DGQTensor({idx_inm1, idx_outm1});
  DGQTensor ten1 = DGQTensor({idx_inp1, idx_outp1});
  ten0(0, 0) = 0.5;
  ten1(0, 0) = 1.2;
  DGQTensor ten2 = DGQTensor({idx_in2, idx_out2});
  ten2(0, 0) = 1.4;
  ten2(1, 1) = 1.0;
  DGQTensor ten3 = DGQTensor({idx_inm1, idx_outm1});
  ten3({0, 0}) = 0.6;
  DGQTensor ten4 = DGQTensor({idx_inm1, idx_outm2});
  ten4({0, 0}) = 1.0;
  ten4({0, 1}) = 1.0;
  DGQTensor ten5 = DGQTensor({idx_in4, idx_out2});
  ten5({0, 0}) = 1.0;
  ten5({1, 0}) = 1.0;
  ten5({2, 1}) = 1.0;
  ten5({3, 1}) = 1.0;
//   RunTestTenMCExpansionCase(ten0, ten1, {0, 1}, ten2);
  RunTestTenMCExpansionCase(ten3, ten3, {1}, ten3);
  RunTestTenMCExpansionCase(ten2, ten2, {0}, ten2);

  DGQTensor ten6 = DGQTensor({idx_in0, idx_out2, idx_out2});
  DGQTensor ten7 = ten6;
  DGQTensor ten8 = DGQTensor({idx_in0_2, idx_out2, idx_out2});
  ten6({0,0,1})=0.3;
  ten7({0,1,0})=0.8;
  ten8({0,0,1})=1;
  ten8({1,1,0})=1;
  RunTestTenMCExpansionCase(ten6,ten7,{0},ten6);

  IndexT idx_in4scts1({qnsct0_2, qnsctp1_2, qnsctp2_2, qnsctm1_2}, IN);
  IndexT idx_in4scts2({qnsctp1_2, qnsctp2_2, qnsctm2_2, qnsctm1_2}, IN);
  IndexT idx_out5scts({qnsct0_2, qnsctp1_2, qnsctp2_2, qnsctm1_2, qnsctm2_2}, OUT);
  DGQTensor ten9({idx_in4scts1, idx_out5scts});
  DGQTensor ten10({idx_in4scts2, idx_out5scts});
  IndexT idx_in5scts({qnsct0_2, qnsctp1_2, qnsctp2_2, qnsctm1_2, qnsctm2_2}, IN);
  DGQTensor ten11({idx_in5scts, idx_out5scts});
  ten9(0, 0) = 1.0;
  ten9(3, 2) = 2.0;
  ten9(7, 6) = 3.0;
  ten10(1, 3) = 4.0;
  ten10(2, 4) = 5.0;
  ten10(5, 8) = 6.0;
  ten11(0, 0) = 1.0;
  ten11(3, 2) = 2.0;
  ten11(7, 6) = 3.0;
  ten11(idx_in4scts1.dim() + 1, 8) = 6.0;
  RunTestTenMCExpansionCase(ten9, ten10, {0}, ten11);
  
}
