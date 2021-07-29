// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-7-29
*
* Description: GraceQ/tensor project.  Unittests for tensor contraction with restriction on 1 sector
*/

#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "gqten/tensor_manipulation/basic_operations.h"     // Dag
#include "gqten/tensor_manipulation/ten_ctrct_1sector.h"
#include "gqten/tensor_manipulation/ten_linear_combine.h"

#include "gtest/gtest.h"
#include "../testing_utility.h"

#include "mkl.h"
#include <iostream>
#include <fstream>
#include "gqten/utility/timer.h"
using namespace gqten;

using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

struct TestContraction : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  int d_s = 3;
  QNSctT qnsct0_s =  QNSctT(qn0,  d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  int d_l = 10;
  QNSctT qnsct0_l =  QNSctT(qn0,  d_l);
  QNSctT qnsctp1_l = QNSctT(qnp1, d_l);
  QNSctT qnsctm1_l = QNSctT(qnm1, d_l);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::OUT);
  IndexT idx_in_l =  IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, GQTenIndexDirType::IN);
  IndexT idx_out_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, GQTenIndexDirType::OUT);

  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_1d_l = DGQTensor({idx_out_l});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_2d_l = DGQTensor({idx_in_l, idx_out_l});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  DGQTensor dten_3d_s3 = DGQTensor({idx_in_s, idx_in_s, idx_out_s});
  DGQTensor dten_3d_s4 = DGQTensor({idx_out_s, idx_in_s, idx_in_s});
  DGQTensor dten_3d_l = DGQTensor({idx_in_l, idx_out_l, idx_out_l});

  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_1d_l = ZGQTensor({idx_out_l});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_2d_l = ZGQTensor({idx_in_l, idx_out_l});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
  ZGQTensor zten_3d_s3 = ZGQTensor({idx_in_s, idx_in_s, idx_out_s});
  ZGQTensor zten_3d_s4 = ZGQTensor({idx_out_s, idx_in_s, idx_in_s});
  ZGQTensor zten_3d_l = ZGQTensor({idx_in_l, idx_out_l, idx_out_l});
};

TEST_F(TestContraction, 2DCase){
  auto dten_2d_s2 = dten_2d_s;
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qn0);

  DGQTensor dres_ten1, dres_ten2, dres_ten3;
  Contract1Sector(&dten_2d_s, 0,0,&dten_2d_s2,{{1},{0}},&dres_ten1);
  Contract1Sector(&dten_2d_s, 0,1,&dten_2d_s2,{{1},{0}},&dres_ten2);
  Contract1Sector(&dten_2d_s, 0,2,&dten_2d_s2,{{1},{0}},&dres_ten3);
//   dres_ten1.Show();
  
  DGQTensor dres_ten_2d_s;
  Contract(&dten_2d_s,&dten_2d_s2,{{1},{0}},&dres_ten_2d_s);
//   dres_ten_2d_s.Show();
  EXPECT_EQ(dres_ten_2d_s, (dres_ten1+dres_ten2+dres_ten3) );
}

TEST_F(TestContraction, 3DCase){
  dten_3d_s.Random(qn0);
  dten_3d_s3.Random(qn0);

  size_t split_idx = 0;
  size_t num_qn = dten_3d_s.GetIndexes()[split_idx].GetQNSctNum();
  std::vector<DGQTensor *> split_tens( num_qn );
  for(size_t i=0;i<num_qn;i++){
      split_tens[i] = new DGQTensor();
      Contract1Sector(&dten_3d_s, split_idx, i ,&dten_3d_s3,{{2},{0}},split_tens[i]);
  }
  std::vector<GQTEN_Double> coefs(num_qn,1.0);
  DGQTensor sum_ten;
  LinearCombine(coefs, split_tens, 0.0, &sum_ten );

  DGQTensor res_ten;
  Contract(&dten_3d_s, &dten_3d_s3,{{2},{0}}, &res_ten);
  EXPECT_EQ(sum_ten, res_ten);

  for(size_t i=0;i<num_qn;i++){
    delete split_tens[i];
  }


  //split for b
  split_idx = 1;
  num_qn = dten_3d_s3.GetIndexes()[split_idx].GetQNSctNum();
  split_tens = std::vector<DGQTensor *>(num_qn);
  for(size_t i=0;i<num_qn;i++){
      split_tens[i] = new DGQTensor();
      Contract1Sector(&dten_3d_s, &dten_3d_s3,split_idx, i ,{{2},{0}},split_tens[i]);
  }
  coefs = std::vector<GQTEN_Double>(num_qn,1.0);
  sum_ten = DGQTensor();
  LinearCombine(coefs, split_tens, 0.0, &sum_ten );

  EXPECT_EQ(sum_ten, res_ten);

  for(size_t i=0;i<num_qn;i++){
    delete split_tens[i];
  }
  
}


TEST(ActualCombat, SSHHubbardD14000){
  using U1U1QN = QN<U1QNVal,U1QNVal>;
  using DGQTensor2 = GQTensor<GQTEN_Double,U1U1QN>;
  std::ifstream ifs("/Users/hxwang/Downloads/mps_ten680.gqten");
  if(!ifs.good()){
    std::cout << "opening file does not success(680)!" <<std::endl;
    exit(1);
  }
  DGQTensor2 mps680;
  ifs >> mps680;
  ifs.close();


  ifs.open("/Users/hxwang/Downloads/mps_ten681.gqten");
  if(!ifs.good()){
    std::cout << "opening file does not success(681)!" <<std::endl;
    exit(1);
  }
  DGQTensor2 mps681;
  ifs >> mps681;
  ifs.close();
  using std::vector;
  vector<vector<size_t>> contract_axes = {{2},{0}};
  DGQTensor2 res;


  std::cout << "\n";
  Timer contract_timer("directly contract");
  Contract(&mps680, &mps681, contract_axes, &res);
  contract_timer.PrintElapsed();
  std::cout << "\n";

  
  Timer contract_split_timer("split contract");
  size_t split_idx = 0;
  size_t num_qn = mps680.GetIndexes()[split_idx].GetQNSctNum();
  std::vector<DGQTensor2 *> split_tens( num_qn );
  for(size_t i=0;i<num_qn;i++){
      split_tens[i] = new DGQTensor2();
      Contract1Sector(&mps680, split_idx, i ,&mps681,contract_axes,split_tens[i]);
  }
  std::vector<GQTEN_Double> coefs(num_qn,1.0);
  DGQTensor2 sum_ten;
  Timer sum_timer("summation");
  LinearCombine(coefs, split_tens, 0.0, &sum_ten);
  sum_timer.PrintElapsed();
  contract_split_timer.PrintElapsed();
  std::cout << "\n";
  EXPECT_EQ(res, sum_ten);

  for(size_t i=0;i<num_qn;i++){
    delete split_tens[i];
  }


}
