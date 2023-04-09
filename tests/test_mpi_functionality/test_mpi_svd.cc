// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang<wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-08-22
* 
* Description: GraceQ/tensor project. Unittests for MPI tensor SVD.
*/


#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_decomp/ten_svd.h"   // SVD
#include "gqten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "gqten/tensor_manipulation/basic_operations.h"     // Dag
#include "gqten/mpi_tensor_manipulation/ten_decomp/mpi_svd.h" //MPISVD
#include "gqten/utility/utils_inl.h"
#include "gqten/framework/hp_numeric/lapack.h"
#include "gqten/utility/timer.h"
#include "gtest/gtest.h"
#include "../testing_utility.h"


#include <vector>
#include <iostream>
#include <fstream>

#include "mkl.h"    // Included after other header file. Because GraceQ needs redefine MKL_Complex16 to gqten::GQTEN_Complex


using namespace gqten;
using namespace std;
using U1U1QN = QN<U1QNVal,U1QNVal>;
using U1QN = QN<U1QNVal>;
using IndexT = Index<U1U1QN>;
using IndexT1 = Index<U1QN>;
using QNSctT = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1U1QN>;
using DGQTensor1 = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor1 = GQTensor<GQTEN_Complex, U1QN>;
const U1QN qn0 = U1QN({QNCard("qn", U1QNVal(0))});
//helper
Index<U1QN> RandIndex(const unsigned qn_sct_num,  //how many quantum number sectors?
                 const unsigned max_dim_in_one_qn_sct, // maximum dimension in every quantum number sector?
                 const GQTenIndexDirType dir){
  QNSectorVec<U1QN> qnsv(qn_sct_num);
  for(size_t i=0;i<qn_sct_num;i++){
    auto qn = U1QN({QNCard("qn", U1QNVal(i))});
    srand((unsigned)time(NULL));
    unsigned degeneracy = rand()%max_dim_in_one_qn_sct+1;
    qnsv[i] = QNSector(qn, degeneracy);
  }
  return Index(qnsv, dir);
}

int main(int argc, char *argv[]) {
  using std::vector;
  namespace mpi = boost::mpi;
  mpi::environment env(mpi::threading::multiple);
  mpi::communicator world;
  size_t thread_num;
  if (argc == 1) {// no input parameter
    thread_num = 12;
  }else{
    thread_num = atoi(argv[1]);
  }
  hp_numeric::SetTensorManipulationThreads(thread_num);
  hp_numeric::SetTensorTransposeNumThreads(thread_num);

  if( world.rank() == kMPIMasterRank){
    if (env.thread_level() < mpi::threading::multiple){
      std::cout << "warning: env.thread_level() < mt::threading::multiple" << std::endl;
    }

    auto index1_in = RandIndex(20,30, gqten::IN);
    auto index2_in = RandIndex(4,5, gqten::IN);
    auto index1_out = RandIndex(4,5, gqten::OUT);
    auto index2_out = RandIndex(20,30, gqten::OUT);
    DGQTensor1 dstate({index2_out,index1_in,index2_in, index1_out});
    dstate.Random(qn0);
    ZGQTensor1 zstate = ToComplex(dstate);
    std::cout << "Randomly generate double and complex tensors." << "\n";

    cout << "Concise Infos of the double tensor: \n";
    dstate.ConciseShow();

    const size_t svd_ldims = 2;
    const U1QN left_div = Div(dstate);
    const GQTEN_Double trunc_err = 1e-8;
    const size_t Dmin = dstate.GetIndexes()[0].dim();
    const size_t Dmax = Dmin;

    DGQTensor1 du,ds,dvt;
    GQTEN_Double actual_trunc_err;
    size_t D;
    Timer parallel_svd_timer("Parallel SVD for the double tensor");
    MPISVDMaster(&dstate,
                 svd_ldims, left_div,
                 trunc_err, Dmin, Dmax,
                 &du, &ds, &dvt, &actual_trunc_err, &D,
                 world
    );
    parallel_svd_timer.PrintElapsed();

    //by single processor svd
    DGQTensor1 du2, ds2, dvt2;
    GQTEN_Double actual_trunc_err2;
    size_t D2;
    Timer single_processor_svd_timer("Single processor SVD for the double tensor");
    SVD(&dstate,
        svd_ldims, left_div,
        trunc_err, Dmin, Dmax,
        &du2, &ds2, &dvt2, &actual_trunc_err2, &D2
    );
    single_processor_svd_timer.PrintElapsed();
    EXPECT_EQ(D, D2);
    EXPECT_NEAR(actual_trunc_err2, actual_trunc_err, 1e-13);
    DGQTensor1 ds_diff = ds+(-ds2);
    EXPECT_NEAR(ds_diff.Normalize()/ds.Normalize(), 0.0, 1e-13 );

    std::cout << "Results from single processor SVD == Results from MPI SVD (double tensor)" << std::endl;

    ZGQTensor1 zu, zvt;
    DGQTensor1 zs;
    Timer parallel_svd_timer_complex("Parallel SVD for the complex tensor");
    MPISVDMaster(&zstate,
                 svd_ldims, left_div,
                 trunc_err, Dmin, Dmax,
                 &zu, &zs, &zvt, &actual_trunc_err, &D,
                 world
    );
    parallel_svd_timer_complex.PrintElapsed();

    //by single processor svd
    ZGQTensor1 zu2, zvt2;
    DGQTensor1 zs2;
    Timer single_processor_svd_timer_complex("Single processor SVD for the complex tensor");
    SVD(&zstate,
        svd_ldims, left_div,
        trunc_err, Dmin, Dmax,
        &zu2, &zs2, &zvt2, &actual_trunc_err2, &D2
    );
    single_processor_svd_timer_complex.PrintElapsed();

    EXPECT_EQ(D, D2);
    EXPECT_NEAR(actual_trunc_err2, actual_trunc_err, 1e-13);
    DGQTensor1 zs_diff = zs + (-zs2);
    EXPECT_NEAR(zs_diff.Normalize() / zs.Normalize(), 0.0, 1e-13);
    std::cout << "Results from single processor SVD == Results from MPI SVD (complex tensor)" << std::endl;

  } else {
    MPISVDSlave<GQTEN_Double>(world);
    MPISVDSlave<GQTEN_Complex>(world);
  }



#ifdef ACTUALCOMBAT
  if( world.rank() == kMPIMasterRank){
    if (env.thread_level() < mpi::threading::multiple){
      std::cout << "warning: env.thread_level() < mt::threading::multiple" << std::endl;
    }
    DGQTensor state;
    std::string file = "state.gqten";

    if( access( file.c_str(), 4) != 0){
        std::cout << "The progress doesn't access to read the file " << file << "!" << std::endl;
        exit(1);
    }
    std::ifstream ifs(file, std::ios::binary);
    if(!ifs.good()){
        std::cout << "The progress can not read the file " << file << " correctly!" << std::endl;
        exit(1);
    }
    ifs >> state;
    std::cout << "The progress has loaded the tensors." <<std::endl;
    cout << "Concise Info of tensors: \n";
    cout << "state.gqten:"; state.ConciseShow();

    //parameter
    const size_t svd_ldims = 2;
    const U1U1QN left_div = Div(state);
    const GQTEN_Double trunc_err = 1e-8;
    const size_t Dmin = state.GetIndexes()[0].dim();
    const size_t Dmax = Dmin;

    //return value;
    DGQTensor u,s,vt;
    GQTEN_Double actual_trunc_err;
    size_t D;
    Timer parallel_svd_timer("parallel svd");
    MPISVDMaster(&state,
          svd_ldims, left_div,
          trunc_err, Dmin, Dmax,
          &u, &s, &vt, &actual_trunc_err, &D,
          world
         );
    parallel_svd_timer.PrintElapsed();
    
    //by single processor svd
    DGQTensor u2, s2, vt2;
    GQTEN_Double actual_trunc_err2;
    size_t D2;
    Timer single_processor_svd_timer("single processor svd");
    SVD(&state,
          svd_ldims, left_div,
          trunc_err, Dmin, Dmax,
          &u2, &s2, &vt2, &actual_trunc_err2, &D2
         );
    single_processor_svd_timer.PrintElapsed();
    EXPECT_EQ(D, D2);
    EXPECT_NEAR(actual_trunc_err2, actual_trunc_err, 1e-13);
    DGQTensor s_diff = s+(-s2);
    EXPECT_NEAR(s_diff.Normalize()/s.Normalize(), 0.0, 1e-13 );


  }else{
    MPISVDSlave<GQTEN_Double>(world);
  }
#endif



  return 0;



}