// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-12-03 21:34
* 
* Description: GraceQ/tensor project. Unittests for tensor SVD.
*/
#include "gqten/gqtensor_all.h"
#include "gqten/tensor_manipulation/ten_decomp/ten_svd.h"   // SVD
#include "gqten/tensor_manipulation/ten_ctrct.h"            // Contract
#include "gqten/tensor_manipulation/basic_operations.h"     // Dag
#include "gqten/utility/utils_inl.h"
#include "gqten/framework/hp_numeric/lapack.h"
#include "gqten/utility/timer.h"
#include "gtest/gtest.h"
#include "../testing_utility.h"

#include "mkl.h"    // Included after other header file. Because GraceQ needs redefine MKL_Complex16 to gqten::GQTEN_Complex
#include <thread>     //hardware_concurrency()

using namespace gqten;
using U1QN = QN<U1QNVal>;
using IndexT = Index<U1QN>;
using QNSctT = QNSector<U1QN>;
using QNSctVecT = QNSectorVec<U1QN>;

using DGQTensor = GQTensor<GQTEN_Double, U1QN>;
using ZGQTensor = GQTensor<GQTEN_Complex, U1QN>;

struct TestSvd : public testing::Test {
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  U1QN qnp1 = U1QN({QNCard(qn_nm, U1QNVal( 1))});
  U1QN qnp2 = U1QN({QNCard(qn_nm, U1QNVal( 2))});
  U1QN qnm1 = U1QN({QNCard(qn_nm, U1QNVal(-1))});
  U1QN qnm2 = U1QN({QNCard(qn_nm, U1QNVal(-2))});

  size_t d_s = 3;
  QNSctT qnsct0_s =  QNSctT(qn0,  d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);

  DGQTensor dten_1d_s = DGQTensor({idx_out_s});
  DGQTensor dten_2d_s = DGQTensor({idx_in_s, idx_out_s});
  DGQTensor dten_3d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s});
  DGQTensor dten_4d_s = DGQTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});

  ZGQTensor zten_1d_s = ZGQTensor({idx_out_s});
  ZGQTensor zten_2d_s = ZGQTensor({idx_in_s, idx_out_s});
  ZGQTensor zten_3d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s});
  ZGQTensor zten_4d_s = ZGQTensor({idx_in_s, idx_out_s, idx_out_s, idx_out_s});
};


inline size_t IntDot(const size_t &size, const size_t *x, const size_t *y) {
  size_t res = 0;
  for (size_t i = 0; i < size; ++i) { res += x[i] * y[i]; }
  return res;
}


inline double ToDouble(const double d) {
  return d;
}


inline double ToDouble(const GQTEN_Complex z) {
  return z.real();
}


inline void SVDTensRestore(
    const DGQTensor *pu,
    const DGQTensor *ps,
    const DGQTensor *pvt,
    const size_t ldims,
    DGQTensor *pres) {
  DGQTensor t_restored_tmp;
  Contract(pu, ps, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}


inline void SVDTensRestore(
    const ZGQTensor *pu,
    const DGQTensor *ps,
    const ZGQTensor *pvt,
    const size_t ldims,
    ZGQTensor *pres) {
  ZGQTensor t_restored_tmp;
  auto zs = ToComplex(*ps);
  Contract(pu, &zs, {{ldims}, {0}}, &t_restored_tmp);
  Contract(&t_restored_tmp, pvt, {{ldims}, {0}}, pres);
}


template <typename TenT>
void CheckIsIdTen(const TenT &t) {
  auto shape = t.GetShape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], shape[1]);
  for (size_t i = 0; i < shape[0]; ++i) {
    GQTEN_Complex elem = t.GetElem({i, i});
    EXPECT_NEAR(elem.real(), 1.0, 1E-14);
    EXPECT_NEAR(elem.imag(), 0.0, 1E-14);
  }
}


template <typename TenElemT, typename QNT>
void RunTestSvdCase(
    GQTensor<TenElemT, QNT> &t,
    const size_t &ldims,
    const size_t &rdims,
    const double &cutoff,
    const size_t &dmin,
    const size_t &dmax,
    const QNT *random_div = nullptr) {
  if (random_div != nullptr) {
    srand(0);
    t.Random(*random_div);
  }
  GQTensor<TenElemT, QNT> u, vt;
  GQTensor<GQTEN_Double, QNT> s;
  double trunc_err;
  size_t D;
  std::string qn_nm = "qn";
  U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
  SVD(
      &t,
      ldims,
      qn0,
      cutoff, dmin, dmax,
      &u, &s,&vt, &trunc_err, &D
  );

  // Canonical check
  GQTensor<TenElemT, QNT> temp1, temp2;
  auto u_dag = Dag(u);
  std::vector<size_t> cano_check_u_ctrct_axes;
  for (size_t i = 0; i < ldims; ++i) { cano_check_u_ctrct_axes.push_back(i); }
  Contract(&u, &u_dag, {cano_check_u_ctrct_axes, cano_check_u_ctrct_axes}, &temp1);
  CheckIsIdTen(temp1);
  auto vt_dag = Dag(vt);
  std::vector<size_t> cano_check_vt_ctrct_axes;
  for (size_t i = 1; i <= rdims; ++i) { cano_check_vt_ctrct_axes.push_back(i); }
  Contract(&vt, &vt_dag, {cano_check_vt_ctrct_axes, cano_check_vt_ctrct_axes}, &temp2);
  CheckIsIdTen(temp2);

  auto ndim = ldims + rdims;
  size_t rows = 1, cols = 1;
  for (size_t i = 0; i < ndim; ++i) {
    if (i < ldims) {
      rows *= t.GetIndexes()[i].dim();
    } else {
      cols *= t.GetIndexes()[i].dim();
    }
  }
  auto dense_mat = new TenElemT [rows*cols];
  auto offsets = CalcMultiDimDataOffsets(t.GetShape());
  for (auto &coors : GenAllCoors(t.GetShape())) {
    dense_mat[IntDot(ndim, coors.data(), offsets.data())] = t.GetElem(coors);
  }
  TenElemT *dense_u;
  TenElemT *dense_vt;
  GQTEN_Double *dense_s;
  hp_numeric::MatSVD(dense_mat, rows, cols, dense_u, dense_s, dense_vt);
  size_t dense_sdim;
  if (rows > cols) {
    dense_sdim = cols;
  } else {
    dense_sdim = rows;
  }

  std::vector<double> dense_svs;
  for (size_t i = 0; i < dense_sdim; ++i) {
    if (dense_s[i] > 1.0E-13) {
      dense_svs.push_back(dense_s[i]);
    }
  }
  std::sort(dense_svs.begin(), dense_svs.end());
  auto endit = dense_svs.cend();
  auto begit =  endit - dmax;
  if (dmax > dense_svs.size()) { begit = dense_svs.cbegin(); }
  auto saved_dense_svs = std::vector<double>(begit, endit);
  std::vector<double> qn_svs;
  for (size_t i = 0; i < s.GetShape()[0]; i++) {
    qn_svs.push_back(ToDouble(s.GetElem({i, i})));
  }
  std::sort(qn_svs.begin(), qn_svs.end());
  EXPECT_EQ(qn_svs.size(), saved_dense_svs.size());
  for (size_t i = 0; i < qn_svs.size(); ++i) {
    EXPECT_NEAR(qn_svs[i], saved_dense_svs[i], kEpsilon);
  }

  double total_square_sum = 0.0;
  for (auto &sv : dense_svs) {
    total_square_sum += sv * sv;
  }
  double saved_square_sum = 0.0;
  for (auto &ssv : saved_dense_svs) {
    saved_square_sum += ssv * ssv;
  }
  auto dense_trunc_err = 1 - saved_square_sum / total_square_sum;
  EXPECT_NEAR(trunc_err, dense_trunc_err, kEpsilon);

  if (trunc_err < 1.0E-10) {
    GQTensor<TenElemT, QNT> t_restored;
    SVDTensRestore(&u, &s, &vt, ldims, &t_restored);
    for (auto &coors : GenAllCoors(t.GetShape())) {
      GtestExpectNear(t_restored.GetElem(coors), t.GetElem(coors), kEpsilon);
    }
  } else {
    GQTensor<TenElemT, QNT> t_restored;
    SVDTensRestore(&u, &s, &vt, ldims, &t_restored);
    auto t_diff = t + (-t_restored);
    auto t_diff_norm = t_diff.Normalize();
    auto t_norm = t.Normalize();
    auto norm_ratio = (t_diff_norm / t_norm);
    GtestExpectNear(norm_ratio * norm_ratio, trunc_err, 1E-02);
  }

  mkl_free_buffers();
  delete [] dense_mat;
  free(dense_s);
  free(dense_u);
  free(dense_vt);
}


TEST_F(TestSvd, 2DCase) {
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qn0);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s - 1,
      &qn0);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s + 1,
      &qn0);

  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s - 1,
      &qnp1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s + 1,
      &qnp1);

  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnp2);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp2);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnm1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm1);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnm2);
  RunTestSvdCase(
      dten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qn0);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnp2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnp2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnm1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm1);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s*3,
      &qnm2);
  RunTestSvdCase(
      zten_2d_s,
      1, 1,
      0, 1, d_s,
      &qnm2);
}


TEST_F(TestSvd, 3DCase) {
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s - 1,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s + 1,
      &qn0);

  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s*2,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s + 1,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      1, 2,
      0, 1, d_s - 1,
      &qnp1);

  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s + 1,
      &qn0);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s - 1,
      &qn0);

  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      dten_3d_s,
      2, 1,
      0, 1, d_s*2,
      &qnp1);

  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      zten_3d_s,
      1, 2,
      0, 1, d_s*2,
      &qnp1);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      zten_3d_s,
      2, 1,
      0, 1, d_s*2,
      &qnp1);
}


TEST_F(TestSvd, 4DCase) {
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3)*(d_s*3),
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3),
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3) + 1,
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3) - 1,
      &qn0);

  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3)*(d_s*3),
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3),
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3) + 1,
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      2, 2,
      0, 1, (d_s*3) - 1,
      &qnp1);

  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      dten_4d_s,
      1, 3,
      0, 1, d_s*2,
      &qnp1);

  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s*3)*(d_s*3),
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s*3),
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s*3)*(d_s*3),
      &qnp1);
  RunTestSvdCase(
      zten_4d_s,
      2, 2,
      0, 1, (d_s*3),
      &qnp1);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s*3,
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s*2,
      &qn0);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s*3,
      &qnp1);
  RunTestSvdCase(
      zten_4d_s,
      1, 3,
      0, 1, d_s*2,
      &qnp1);
}


struct TestSvdOmpParallel : public testing::Test {
  std::string qn_nm = "qn";
};

IndexT RandIndex(const unsigned qn_sct_num,  //how many quantum number sectors?
                 const unsigned max_dim_in_one_qn_sct, // maximum dimension in every quantum number sector?
                 const GQTenIndexDirType dir){
    QNSectorVec<U1QN> qnsv(qn_sct_num);
    for(size_t i=0;i<qn_sct_num;i++){
        auto qn = U1QN({QNCard("qn", U1QNVal(i))});
        srand(i*i/3);
        unsigned degeneracy = rand()%max_dim_in_one_qn_sct+1;
        qnsv[i] = QNSector(qn, degeneracy);
    }
    return Index(qnsv, dir);
}

// template <typename TenElemT, typename QNT>
void ConciseShow(DGQTensor& t){
    using std::cout;
    using std::endl;
    cout << "  Tensor Concise Info: " <<"\n";
    ShapeT shape = t.GetShape();
    cout << "\t tensor shape: " << "\t[";
    for(size_t i = 0;i<shape.size();i++){
        if(i<shape.size()-1){
            cout << shape[i] << ",  ";
        }else{
            cout << shape[i] <<"]\n";
        }
    }
    cout << "\t tensor elementary type: " << "GQTEN_Double";
    cout << "\t tensor qn block: " << t.GetQNBlkNum() << "\n";
    unsigned total_size = t.size();
    unsigned data_size = t.GetBlkSparDataTen().GetActualRawDataSize();
    cout << "\t tensor size(product of shape):" << total_size<<"\n";
    cout << "\t actual data size: " << data_size <<"\n";
    cout << "\t tensor sparsity: " << double(data_size) / double(total_size) << endl;
}

template <typename TenElemT, typename QNT>
void RunTestSvdOmpCase(DGQTensor& t,//Only index
    const size_t &ldims,
    const size_t &rdims,
    const double &cutoff,
    const size_t &dmin,
    const size_t &dmax,
    const size_t &omp_outer_th){
    srand(3);
    std::string qn_nm = "qn";
    U1QN qn0 =  U1QN({QNCard(qn_nm, U1QNVal( 0))});
    t.Random(qn0);

    
    ConciseShow(t);
    // unsigned max_thread = std::thread::hardware_concurrency();
    unsigned max_thread = 20;

    Timer single_thread_timer("svd_single_thread");
    gqten::hp_numeric::SetTensorManipulationTotalThreads(max_thread);
    gqten::hp_numeric::SetTensorDecompOuterParallelThreads(1);

    GQTensor<TenElemT, QNT> u1, vt1;
    GQTensor<GQTEN_Double, QNT> s1;
    double trunc_err1;
    size_t D1;

    SVD(
        &t,
        ldims,
        qn0,
        cutoff, dmin, dmax,
        &u1, &s1,&vt1, &trunc_err1, &D1
    );

    single_thread_timer.PrintElapsed();

    Timer nested_thread_timer("svd_nested_omp_thread");
    assert(omp_outer_th>1);
    gqten::hp_numeric::SetTensorManipulationTotalThreads(max_thread);
    gqten::hp_numeric::SetTensorDecompOuterParallelThreads(omp_outer_th);
    GQTensor<TenElemT, QNT> u2, vt2;
    GQTensor<GQTEN_Double, QNT> s2;
    double trunc_err2;
    size_t D2;
    SVD(
        &t,
        ldims,
        qn0,
        cutoff, dmin, dmax,
        &u2, &s2,&vt2, &trunc_err2, &D2
    );
    nested_thread_timer.PrintElapsed();

    EXPECT_NEAR(trunc_err1, trunc_err2, kEpsilon);
    EXPECT_EQ(D1,D2);
    EXPECT_EQ(u1, u2);
    EXPECT_EQ(s1.GetShape(), s2.GetShape());
    for(size_t i=0;i<s1.GetShape()[0];i++ ){
        EXPECT_NEAR(s1({i,i}),s2({i,i}),kEpsilon*s1.GetShape()[0]);
    }
    EXPECT_EQ(vt1,vt2);

}

TEST(bench_mark_for_nested_omp_parallel, 2Dcase){
    auto index1_in = RandIndex(5,4,gqten::IN);
    auto index1_out = RandIndex(5,4, gqten::OUT);
    DGQTensor t1({index1_in,index1_out});

    RunTestSvdOmpCase<GQTEN_Double,U1QN>(t1, size_t(1), size_t(1),
                 1e-8,size_t(10),size_t(10),size_t(2));



    auto index2_in = RandIndex(200,1200,gqten::IN);
    auto index2_out = RandIndex(200,1200, gqten::OUT);
    DGQTensor t2({index2_in,index2_out});

    RunTestSvdOmpCase<GQTEN_Double,U1QN>(t2, size_t(1), size_t(1),
                 1e-8,size_t(30),size_t(30),10);
}
