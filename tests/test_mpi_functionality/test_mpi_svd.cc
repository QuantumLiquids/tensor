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
using IndexT = Index<U1U1QN>;
using QNSctT = QNSector<U1U1QN>;
using QNSctVecT = QNSectorVec<U1U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1U1QN>;


int main(int argc, char *argv[]){
  namespace mpi = boost::mpi;
  using std::vector;
  mpi::environment env(mpi::threading::multiple);
  mpi::communicator world;
  size_t thread_num;
  if(argc == 1){// no input paramter
    thread_num = 12;
  }else{
    thread_num = atoi(argv[1]);
  }
  hp_numeric::SetTensorManipulationTotalThreads(thread_num);
  hp_numeric::SetTensorTransposeNumThreads(thread_num);

  if( world.rank() == kMPIMasterRank){
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

    //paramter
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

  return 0;



}