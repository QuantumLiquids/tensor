// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-7-26
*
* Description: GraceQ/tensor project. Unittests for Boost MPI of GQTensor.
*/

//Note run this executable as: /usr/bin/mpirun -np 2 ./tests/test_boost_mpi
#include "gqten/gqten.h"      // GQTensor, Index, QN, U1QNVal, QNSectorVec
#include "gqten/utility/utils_inl.h"        // GenAllCoors
#include "gqten/utility/timer.h"
#include "gtest/gtest.h"
#include "../testing_utility.h"     // RandInt, RandUnsignedInt, TransCoors

#include <fstream>    // ifstream, ofstream
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include "mpi.h"
namespace mpi = boost::mpi;


using namespace gqten;

using U1QN = QN<U1QNVal>;
using U1U1QN = QN<U1QNVal, U1QNVal>;
using DGQTensor2 = GQTensor<GQTEN_Double,U1U1QN>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

const std::string qn_nm = "qn_nm";
U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
namespace TestGQTensor{
  std::string qn_nm = "qn";
  
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  QNSctT qnsct0_s =  QNSctT(qn0,  4);
  QNSctT qnsctp1_s = QNSctT(qnp1, 5);
  QNSctT qnsctm1_s = QNSctT(qnm1, 3);
  QNSctT qnsct0_l =  QNSctT(qn0,  10);
  QNSctT qnsctp1_l = QNSctT(qnp1, 8);
  QNSctT qnsctm1_l = QNSctT(qnm1, 12);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, GQTenIndexDirType::OUT);
  IndexT idx_in_l =  IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, GQTenIndexDirType::IN);
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


//helper
IndexT RandIndex(const unsigned qn_sct_num,  //how many quantum number sectors?
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


int main(int argc, char* argv[])
{
  mpi::environment env;
  mpi::communicator world;
  const size_t num_data = 1e7;
  double* B = new double[num_data];
  if (world.rank() == 0) {
    std::cout << "Test transfers " << num_data <<" double array efficiency by boost." << "\n";
    std::cout << "(The efficiency should almost the same with C API MPI." << std::endl;
    Timer transfer_array_boost("send_and_recv_array_by_boost");
    world.send(1, 0, B, num_data);
    world.recv(1, 2, B, num_data);
    transfer_array_boost.PrintElapsed();
  } else {
    world.recv(0, 0, B, num_data);
    world.send(0, 2, B, num_data);
  }
  delete[] B;

  if(world.rank() == 0){
    std::cout << "Test for random tensor." <<std::endl;
    auto index1_in = RandIndex(50,400, gqten::IN);
    auto index2_in = RandIndex(4,1, gqten::IN);
    auto index1_out = RandIndex(4,1, gqten::OUT);
    auto index2_out = RandIndex(50,400, gqten::OUT);

    // auto index1_in = RandIndex(1,40, gqten::IN);
    // auto index2_in = RandIndex(1,4, gqten::IN);
    // auto index1_out = RandIndex(1,2, gqten::OUT);
    // auto index2_out = RandIndex(1,20, gqten::OUT);

    DGQTensor t1({index2_out,index1_in,index2_in, index1_out});
    t1.Random(qn0);


    t1.ConciseShow();

    Timer mpi_double_transf_timer("mpi_send_recv_send");
    send_gqten(world,1, //to process 1
    35,//tag 
    t1);
    
    DGQTensor t3;// will receive the t2 in process1
    recv_gqten(world,1,// from process1
    37,//tag 
    t3);

    mpi_double_transf_timer.PrintElapsed();
    assert(t1==t3);
    std::cout << "Send-Receive-Send For Random Tensor Success." << std::endl;
  }else {
    DGQTensor t2;

    recv_gqten(world,0,// from process1
    35,//tag 
    t2);

    send_gqten(world,0,
    37,
    t2);
  }


  std::string file="mps_ten987.gqten";
  const std::string FAIL_SIGNAL = "open file " + file + " fail.";
  if (world.rank() == 0) {
    std::cout << "Test for loaded mps tensor" <<std::endl;
    DGQTensor2 t1;
    std::ifstream ifs(file, std::ifstream::binary);
    if(!ifs.good()){
      world.send(1, 235, std::string(FAIL_SIGNAL));
    }else{
      ifs >> t1;
      world.send(1, 235, std::string("read tensor success!"));

      t1.ConciseShow();

      Timer mpi_double_transf_timer("mpi_send_recv_send");
      send_gqten(world,1, //to process 1
      0,//tag 
      t1);
    
      DGQTensor2 t3;// will receive the t2 in process1
      recv_gqten(world,1,// from process1
      3,//tag 
      t3);

      mpi_double_transf_timer.PrintElapsed();
      assert(t1==t3);
      std::cout << "Send-Receive-Send For Tensor In File " 
              << file << " Success." << std::endl;
    }

  } else {
    DGQTensor2 t2;

    std::string msg;
    world.recv(0, 235, msg);
    std::cout << msg << std::endl;

    if(msg != FAIL_SIGNAL){
      recv_gqten(world,0,// from process1
      0,//tag 
      t2);

      send_gqten(world,0,
      3,
      t2);
    }
  }
  return 0;
}