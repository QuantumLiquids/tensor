// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-7-26
*
* Description: GraceQ/tensor project. Unittests for Boost MPI of GQTensor.
*/

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


// TEST_F(TestGQTensor, TestSerializationRandomTensor){
//     std::ofstream ofs("dten_default.gqten");
//     boost::archive::binary_oarchive oa(ofs);

//     auto index1_in = RandIndex(5,4, gqten::IN);
//     auto index2_in = RandIndex(4,6, gqten::IN);
//     auto index1_out = RandIndex(3,3, gqten::OUT);
//     auto index2_out = RandIndex(2,5, gqten::OUT);

//     DGQTensor t1({index2_out,index1_in,index2_in, index1_out});
//     t1.Random(qn0);
//     t1.ConciseShow();
//     oa << t1;
//     ofs.close();
    

//     std::ifstream ifs("dten_default.gqten");
//     boost::archive::binary_iarchive ia(ifs);

//     DGQTensor t2;
//     ia >> t2;
//     ifs.close();
//     EXPECT_EQ(t1, t2);
// }


int main(int argc, char* argv[])
{
  
  // const size_t num_data = 1e7;

/*
  MPI_Init(&argc, &argv);
  int rank;
  double* A = new double[num_data];
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    Timer transfer_array_MPI("transfer_array_by_MPI");
    int result = MPI_Send(A, num_data, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    if (result == MPI_SUCCESS)
      std::cout << "Rank 0 OK!" << std::endl;
    result = MPI_Recv(A, num_data, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD,
			  MPI_STATUS_IGNORE);
    if (result == MPI_SUCCESS)
      std::cout << "Rank 0 OK!" << std::endl;
    transfer_array_MPI.PrintElapsed();
  } else if (rank == 1) {
    int result = MPI_Recv(A, num_data, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
			  MPI_STATUS_IGNORE);
    if (result == MPI_SUCCESS)
      std::cout << "Rank 1 OK!" << std::endl;

    result = MPI_Send(A, num_data, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    if (result == MPI_SUCCESS)
      std::cout << "Rank 1 OK!" << std::endl;
  }
  delete[] A;

  MPI_Finalize();
*/

/*
  mpi::environment env;
  mpi::communicator world;
  double* B = new double[num_data];
  // auto B = boost::serialization::make_array(A, num_data);
  if (world.rank() == 0) {
    Timer transfer_array_boost("transfer_array_by_boost");
    
    world.send(1, 0, B, num_data);
    world.recv(1, 2, B, num_data);
    transfer_array_boost.PrintElapsed();
  } else {
    world.recv(0, 0, B, num_data);
    world.send(0, 2, B, num_data);
  }
  delete[] B;




  return 0;
*/
  mpi::environment env;
  mpi::communicator world;
  if (world.rank() == 0) {
    
    // auto index1_in = RandIndex(50,400, gqten::IN);
    // auto index2_in = RandIndex(4,1, gqten::IN);
    // auto index1_out = RandIndex(4,1, gqten::OUT);
    // auto index2_out = RandIndex(50,400, gqten::OUT);

    // auto index1_in = RandIndex(1,40, gqten::IN);
    // auto index2_in = RandIndex(1,4, gqten::IN);
    // auto index1_out = RandIndex(1,2, gqten::OUT);
    // auto index2_out = RandIndex(1,20, gqten::OUT);

    // DGQTensor t1({index2_out,index1_in,index2_in, index1_out});
    // t1.Random(qn0);
    DGQTensor t1;
    std::ifstream ifs("/Users/hxwang/Documents/mps_ten1.gqten", std::ifstream::binary);
    if(ifs.good()){
      std::cout << "open file success" << std::endl;
    }else{
      std::cout << "open file fail" << std::endl;
      exit(0);
    }
    ifs >> t1;
    std::cout << "load success" << std::endl;
    // world.send(1, 235, std::string("load_finished!"));

    t1.ConciseShow();

    Timer mpi_double_transf_timer("mpi_send_recv_send");
    send_gqten(world,1, //to process 1
    0,//tag 
    t1);
    
    DGQTensor t3;// will receive the t2 in process1
    recv_gqten<GQTEN_Double,U1QN>(world,1,// from process1
    3,//tag 
    t3);

    mpi_double_transf_timer.PrintElapsed();
    // t3.ConciseShow();
    if(t1==t3){
      std::cout << "Send-Receive-Send Success." << std::endl;
    }else{
      // assert(t1==t3);
    }

  } else {
    DGQTensor t2;

    // std::string msg;
    // world.recv(0, 235, msg);
    // std::cout << msg << std::endl;

    recv_gqten<GQTEN_Double,U1QN>(world,0,// from process1
     0,//tag 
    t2);
    t2.ConciseShow();

    send_gqten(world,0,
    3,
    t2);

    // Timer write_read_disk_timer("write_read_disk");
    // std::ofstream ofs("A.gqten");
    // ofs << t2;
    // ofs.close();

    // std::ifstream ifs("A.gqten");
    // DGQTensor t3;
    // ifs >> t3;
    // write_read_disk_timer.PrintElapsed();
  }


}