// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 14:33
* 
* Description: GraceQ/tensor project. The main header file.
*/
#ifndef GQTEN_GQTEN_H
#define GQTEN_GQTEN_H


#include <string>
#include <vector>
#include <fstream>

#include <cmath>

#ifdef GQTEN_MPI_PARALLEL
#include "mpi.h"
#endif


namespace gqten {


// GQTensor storage file suffix.
const std::string kGQTenFileSuffix = "gqten";
// Double numerical error.
const double kDoubleEpsilon = 1.0E-15;
// Default tensor transpose threads number.
const int kTensorTransposeDefaultNumThreads = 4;


// Quantum number.
struct QNNameVal {
  QNNameVal() = default;
  QNNameVal(const std::string &nm, const long val): name(nm), val(val) {}

  std::string name;
  long val;
};

class QN {
friend std::ifstream &bfread(std::ifstream &, QN &);
friend std::ofstream &bfwrite(std::ofstream &, const QN &);

public:
  QN(void);
  QN(const std::vector<QNNameVal> &);
  QN(const std::vector<long> &);

  QN(const QN &);
  QN &operator=(const QN &);

  std::size_t Hash(void) const;

  QN operator-(void) const;
  QN &operator+=(const QN &);

private:
  std::vector<long> values_;
  std::size_t hash_;

  std::size_t CalcHash(void) const;
};

QN operator+(const QN &, const QN &);

QN operator-(const QN &, const QN &);

bool operator==(const QN &, const QN &);

bool operator!=(const QN &, const QN &);

std::ifstream &bfread(std::ifstream &, QN &);

std::ofstream &bfwrite(std::ofstream &, const QN &);


// Quantum number sector.
class QNSector {
friend std::ifstream &bfread(std::ifstream &, QNSector &);
friend std::ofstream &bfwrite(std::ofstream &, const QNSector &);

public:
  QNSector(const QN &qn, const long dim) : qn(qn), dim(dim) {
    hash_ = CalcHash();
  }
  QNSector(void) : QNSector(QN(), 0) {}

  QNSector(const QNSector &qns) : qn(qns.qn), dim(qns.dim), hash_(qns.hash_) {}
  QNSector &operator=(const QNSector &rhs);

  size_t Hash(void) const { return hash_; }

  QN qn;
  long dim;

private:
  size_t CalcHash(void) const { return qn.Hash() ^ dim; }
  size_t hash_;
};

bool operator==(const QNSector &, const QNSector &);

bool operator!=(const QNSector &, const QNSector &);

std::ifstream &bfread(std::ifstream &, QNSector &);

std::ofstream &bfwrite(std::ofstream &, const QNSector &);


// Quantum number sector set.
class QNSectorSet {
public:
  QNSectorSet(void) {}
  QNSectorSet(const std::vector<QNSector> &qnscts) : qnscts(qnscts) {}
  QNSectorSet(const std::vector<const QNSector*> &);

  QNSectorSet(const QNSectorSet &qnss) : qnscts(qnss.qnscts) {}

  virtual ~QNSectorSet() = default;

  virtual size_t Hash(void) const;

  std::vector<QNSector> qnscts;
};

bool operator==(const QNSectorSet &, const QNSectorSet &);

bool operator!=(const QNSectorSet &, const QNSectorSet &);


// Index.
#define NDIR "NDIR"
#define IN "IN"
#define OUT "OUT"

struct InterOffsetQnsct {
  InterOffsetQnsct(const long &inter_offset, const QNSector &qnsct) :
      inter_offset(inter_offset), qnsct(qnsct) {}
  long inter_offset;
  QNSector qnsct;
};

class Index : public QNSectorSet {
friend std::ifstream &bfread(std::ifstream &, Index &);
friend std::ofstream &bfwrite(std::ofstream &, const Index &);

public:
  Index(void) : QNSectorSet(), dim(0), dir(NDIR), tag("") {}

  Index(
      const std::vector<QNSector> &qnscts,
      const std::string &dir,
      const std::string &tag) : QNSectorSet(qnscts), dir(dir), tag(tag) {
    dim = CalcDim(); 
  }
  Index(const std::vector<QNSector> &qnscts) : Index(qnscts, NDIR, "") {}
  Index(const std::vector<QNSector> &qnscts, const std::string &dir) :
      Index(qnscts, dir, "") {}

  Index(const Index &index) :
      QNSectorSet(index.qnscts),
      dim(index.dim), dir(index.dir), tag(index.tag) {}
  Index &operator=(const Index &rhs) {
    qnscts = rhs.qnscts;
    dim = rhs.dim;
    dir = rhs.dir;
    tag = rhs.tag;
    return *this;
  }

  size_t Hash(void) const override;
  InterOffsetQnsct CoorInterOffsetAndQnsct(const long) const;

  // Inplace operations.
  void Dag(void) {
    if (dir == IN) {
      dir = OUT;
    } else if (dir == OUT) {
      dir = IN;
    }
  }

  // Operators overloading.
  bool operator==(const Index &rhs) const { return  Hash() ==  rhs.Hash(); }

  long CalcDim(void) {
    long dim = 0;
    for (auto &qnsct : qnscts) {
      dim += qnsct.dim;
    }
    return dim;
  }

  long dim;
  std::string dir;
  std::string tag;
};

std::ifstream &bfread(std::ifstream &, Index &);

std::ofstream &bfwrite(std::ofstream &, const Index &);


// Dense block labeled by the quantum number.
class QNBlock : public QNSectorSet {
// Binary I/O.
friend std::ifstream &bfread(std::ifstream &, QNBlock &);
friend std::ofstream &bfwrite(std::ofstream &, const QNBlock &);
// Some functions called by tensor numerical functions to use the private constructor.
friend std::vector<QNBlock *> BlocksCtrctBatch(
    const std::vector<long> &, const std::vector<long> &,
    const double,
    const std::vector<QNBlock *> &, const std::vector<QNBlock *> &);
#ifdef GQTEN_MPI_PARALLEL
friend std::vector<QNBlock *> GQTEN_MPI_BlocksCtrctBatch(
    const std::vector<long> &, const std::vector<long> &,
    const double,
    const std::vector<QNBlock *> &, const std::vector<QNBlock *> &,
    MPI_Comm, const int);
#endif


public:
  QNBlock(void) = default;
  QNBlock(const std::vector<QNSector> &);
  QNBlock(const QNBlock &);
  QNBlock &operator=(const QNBlock &);
  
  /* TODO: Moving constructor. */

  ~QNBlock(void) override;
  
  // Element getter and setter.
  const double &operator()(const std::vector<long> &) const;
  double &operator()(const std::vector<long> &);

  // Data access.
  const double *cdata(void) const { return data_; }   // constant reference.
  double * &data(void) { return data_; }              // non-constant reference.

  // Hash methods.
  size_t PartHash(const std::vector<long> &) const;
  size_t QNSectorSetHash(void) const { return qnscts_hash_; }

  // Inplace operations.
  void Random(void);
  void Transpose(const std::vector<long> &);

  // Public data members.
  long ndim = 0;              // Number of dimensions.
  std::vector<long> shape;    // Shape of the block.
  long size = 0;              // Total number of elements in this block.

private:
  // NOTE: For performance reason, this constructor will NOT initialize the data_ to 0!!!
  // It should only be intra-used.
  QNBlock(const std::vector<const QNSector *> &);

  double *data_ = nullptr;    // Data in a 1D array.
  std::vector<long> data_offsets_;
  std::size_t qnscts_hash_ = 0;
};

std::ifstream &bfread(std::ifstream &, QNBlock &);

std::ofstream &bfwrite(std::ofstream &, const QNBlock &);


// Tensor with U1 symmetry.
struct BlkInterOffsetsAndQNSS {     // QNSS: QNSectorSet.
  BlkInterOffsetsAndQNSS(
      const std::vector<long> &blk_inter_offsets, const QNSectorSet &blk_qnss) :
      blk_inter_offsets(blk_inter_offsets), blk_qnss(blk_qnss) {}

  std::vector<long> blk_inter_offsets;
  QNSectorSet blk_qnss;
};

class GQTensor {
friend std::ifstream &bfread(std::ifstream &, GQTensor &);
friend std::ofstream &bfwrite(std::ofstream &, const GQTensor &);

public:
  GQTensor(void) = default;
  GQTensor(const std::vector<Index> &);

  GQTensor(const GQTensor &);
  GQTensor &operator=(const GQTensor &);

  /* TODO: Moving constructor. */

  ~GQTensor(void);

  // Element getter and setter.
  double Elem(const std::vector<long> &) const;     // Getter.
  double &operator()(const std::vector<long> &);    // Setter.

  // Access to the blocks.
  const std::vector<QNBlock *> &cblocks(void) const { return blocks_; }
  std::vector<QNBlock *> &blocks(void) { return blocks_; }

  // Inplace operations.

  // Random set tensor elements with given quantum number divergence.
  // Any original blocks will be destroyed.
  void Random(const QN &);

  // Tensor transpose.
  void Transpose(const std::vector<long> &);

  // Normalize the GQTensor and return its norm.
  double Normalize(void);

  // Switch the direction of the indexes, complex conjugate of the element in the future.
  void Dag(void) { for (auto &index : indexes) { index.Dag(); } }

  // Operators overload.
  GQTensor operator-(void) const;
  GQTensor operator+(const GQTensor &);
  GQTensor &operator+=(const GQTensor &);

  bool operator==(const GQTensor &) const;
  bool operator!=(const GQTensor &rhs) const { return !(*this == rhs); }

  // Iterators.
  // Return all the tensor coordinates. So heavy that you should not use it!
  std::vector<std::vector<long>> CoorsIter(void) const;

  // Public data members.
  std::vector<Index> indexes;
  double scalar = 0.0;
  std::vector<long> shape;

private:
  std::vector<QNBlock *> blocks_;

  double Norm(void);

  BlkInterOffsetsAndQNSS CalcTargetBlkInterOffsetsAndQNSS(
      const std::vector<long> &) const;
  std::vector<QNSectorSet> BlkQNSSsIter(void) const;
};


// GQTensor objects operations.
// For Index.
Index InverseIndex(const Index &);

// For GQTensor.
GQTensor Dag(const GQTensor &);

// Just mock the dag. Not construct a new object.
inline const GQTensor &MockDag(const GQTensor &t) { return t; }

QN Div(const GQTensor &);

GQTensor operator*(const GQTensor &, const double &);

GQTensor operator*(const double &, const GQTensor &);

// GQTensor I/O
std::ifstream &bfread(std::ifstream &, GQTensor &);

std::ofstream &bfwrite(std::ofstream &, const GQTensor &);


// Tensor numerical functions.
// Tensors contraction.
GQTensor *Contract(
    const GQTensor &, const GQTensor &,
    const std::vector<std::vector<long>> &);

#ifdef GQTEN_MPI_PARALLEL
const char kGemmWorkerStatCont = 'c';
const char kGemmWorkerStatStop = 's';


GQTensor *GQTEN_MPI_Contract(
    const GQTensor &, const GQTensor &,
    const std::vector<std::vector<long>> &,
    MPI_Comm, const int);


inline void MPI_SendGemmWorkerStat(
    const char stat, const int worker, MPI_Comm comm) {
  MPI_Send(&stat, 1, MPI_CHAR, worker, 5, comm);
}
#endif

// Tensors linear combination.
// Do the operation: res += (coefs[0]*ts[0] + coefs[1]*ts[1] + ...).
/* TODO: For scalar tensor case. */
void LinearCombine(
    const std::vector<double> &,
    const std::vector<GQTensor *> &,
    GQTensor *);

void LinearCombine(
    const std::size_t,
    const double *,
    const std::vector<GQTensor *> &,
    GQTensor *);

// Tensor SVD.
struct SvdRes {
  SvdRes(
      GQTensor *u, GQTensor *s, GQTensor *v,
      const double trunc_err, const long D) :
      u(u), s(s), v(v), trunc_err(trunc_err), D(D) {}
  GQTensor *u;
  GQTensor *s;
  GQTensor *v;
  const double trunc_err;
  const long D;
};

SvdRes Svd(
    const GQTensor &,
    const long, const long,
    const QN &, const QN &);

SvdRes Svd(
    const GQTensor &,
    const long, const long,
    const QN &, const QN &,
    const double, const long, const long);


// Tensor transpose function multi-thread controller.
int GQTenGetTensorTransposeNumThreads(void);

void GQTenSetTensorTransposeNumThreads(const int);


// Timer.
class Timer {
public:
  Timer(const std::string &);

  void Restart(void);
  double Elapsed(void);
  double PrintElapsed(std::size_t precision = 5);

private:
  double start_;
  std::string notes_;

  double GetWallTime(void);
};
} /* gqten */ 
#endif /* ifndef GQTEN_GQTEN_H */
