// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang<wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-22
* 
* Description: GraceQ/tensor project. Unittests for MPI tensor SVD.
*/

#include "gtest/gtest.h"
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_decomp/ten_svd.h"   // SVD
#include "gqten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "gqten/tensor_manipulation/basic_operations.h"     // Dag
#include "gqten/mpi_tensor_manipulation/ten_decomp/mpi_svd.h" //MPISVD
#include "gqten/utility/utils_inl.h"
#include "gqten/framework/hp_numeric/lapack.h"
#include "gqten/utility/timer.h"
#include "../testing_utility.h"

#include <vector>
#include <iostream>
#include <fstream>

using namespace gqten;
using namespace std;

using U1QN = special_qn::U1QN;
using U1U1QN = special_qn::U1U1QN;

using IndexT1 = Index<U1QN>;
using IndexT = Index<U1U1QN>;

using QNSctT = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1U1QN>;

using DGQTensor1 = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor1 = GQTensor<GQTEN_Complex, U1QN>;

using DGQTensor2 = GQTensor<GQTEN_Double, U1U1QN>;

//helper
Index<U1QN> RandIndex(const unsigned qn_sct_num,  //how many quantum number sectors?
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

namespace mpi = boost::mpi;

struct TestMPISvd : public testing::Test {
  const U1QN qn0 = U1QN(0);

  boost::mpi::communicator world;

  DGQTensor1 dstate;
  ZGQTensor1 zstate;
  void SetUp(void) {
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
      delete listeners.Release(listeners.default_result_printer());
    }
    if (world.rank() == kMPIMasterRank) {
      auto index1_in = RandIndex(20, 30, gqten::IN);
      auto index2_in = RandIndex(4, 5, gqten::IN);
      auto index1_out = RandIndex(4, 5, gqten::OUT);
      auto index2_out = RandIndex(20, 30, gqten::OUT);
      dstate = DGQTensor1({index2_out, index1_in, index2_in, index1_out});
      dstate.Random(qn0);
      zstate = ToComplex(dstate);

      std::cout << "Randomly generate double and complex tensors." << "\n";
      cout << "Concise Infos of the double tensor: \n";
      dstate.ConciseShow();
    }
  }
};

template<typename TenElemT, typename QNT>
void RunTestSvdCase(
    GQTensor<TenElemT, QNT> &t,
    const size_t &svd_ldims,
    const QNT left_div,
    const double &trunc_err,
    const size_t &dmin,
    const size_t &dmax,
    const mpi::communicator &world) {
  using Tensor = GQTensor<TenElemT, QNT>;
  using DTensor = GQTensor<GQTEN_Double, QNT>;
  if (world.rank() == kMPIMasterRank) {
    Tensor u, vt, u2, vt2;
    DTensor s, s2;
    double actual_trunc_err, actual_trunc_err2;
    size_t D, D2;

    Timer parallel_svd_timer("Parallel SVD");
    MPISVDMaster(&t,
                 svd_ldims, left_div,
                 trunc_err, dmin, dmax,
                 &u, &s, &vt, &actual_trunc_err, &D,
                 world
    );
    parallel_svd_timer.PrintElapsed();

    Timer single_processor_svd_timer("Single processor SVD");
    SVD(&t,
        svd_ldims, left_div,
        trunc_err, dmin, dmax,
        &u2, &s2, &vt2, &actual_trunc_err2, &D2
    );
    single_processor_svd_timer.PrintElapsed();
    EXPECT_EQ(D, D2);
    EXPECT_NEAR(actual_trunc_err2, actual_trunc_err, 1e-13);
    DTensor ds_diff = s + (-s2);
    EXPECT_NEAR(ds_diff.Normalize() / s.Normalize(), 0.0, 1e-13);
  } else {
    MPISVDSlave<TenElemT>(world);
  }
}

TEST_F(TestMPISvd, SVD_RandState) {
  size_t dmax = 1;
  if (world.rank() == kMPIMasterRank) {
    dmax = dstate.GetIndexes()[0].dim();
  }
  RunTestSvdCase(dstate, 2, qn0, 1e-8, 1, dmax, world);
  RunTestSvdCase(zstate, 2, qn0, 1e-8, 1, dmax, world);
#ifdef ACTUALCOMBAT
  DGQTensor2 state_load;
  if (world.rank() == kMPIMasterRank) {
    std::string file = "state_load.gqten";
    if (access(file.c_str(), 4) != 0) {
      std::cout << "The progress doesn't access to read the file " << file << "!" << std::endl;
      exit(1);
    }
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs.good()) {
      std::cout << "The progress can not read the file " << file << " correctly!" << std::endl;
      exit(1);
    }
    ifs >> state_load;
    std::cout << "The progress has loaded the tensors." << std::endl;
    cout << "Concise Info of tensors: \n";
    cout << "state_load.gqten:";
    state_load.ConciseShow();
  }
  if (world.rank() == kMPIMasterRank) {
    dmax = state_load.GetIndexes()[0].dim();
  }
  RunTestSvdCase(state_load, 2, U1U1QN(0, 0), 1e-8, 1, dmax, world);
#endif

}

int main(int argc, char *argv[]) {
  int result = 0;
  ::testing::InitGoogleTest(&argc, argv);
  boost::mpi::environment env;
  size_t thread_num;
  if (argc == 1) {// no input parameter
    thread_num = 12;
  } else {
    thread_num = atoi(argv[1]);
  }
  hp_numeric::SetTensorManipulationThreads(thread_num);
  hp_numeric::SetTensorTransposeNumThreads(thread_num);
  result = RUN_ALL_TESTS();
  return result;
}