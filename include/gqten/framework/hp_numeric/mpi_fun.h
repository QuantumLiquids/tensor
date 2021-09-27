// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-09-27 12:34
*
* Description: GraceQ/tensor project. MPI related
*/

/**
@file blas_level1.h
@brief MPI related functions.
*/

#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_MPI_FUN_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_MPI_FUN_H

#include "gqten/framework/value_t.h"      // GQTEN_Double, GQTEN_Complex
#include "mpi.h"                          // MPI_Send, MPI_Recv...

namespace gqten {
/// High performance numerical functions.
namespace hp_numeric {
//block point-to-point communication wrapper
inline void MPI_Send(const GQTEN_Double* data,
  const size_t data_size,
  const int dest,
  const int tag,
  const MPI_Comm& comm
){
  MPI_Send((const void*)data, data_size, MPI_DOUBLE, dest, tag, comm);
}

inline void MPI_Send(const GQTEN_Complex* data,
  const size_t data_size,
  const int dest,
  const int tag,
  const MPI_Comm& comm
){
  MPI_Send((const void*)data, data_size, MPI_CXX_DOUBLE_COMPLEX, dest, tag, comm);
}

// Assum source != mpi_any_source
inline void MPI_Recv(GQTEN_Double* data,
  const size_t data_size,
  const int source,
  const int tag,
  const MPI_Comm& comm
){
  MPI_Status status;
  MPI_Recv( (void*)data, data_size, MPI_DOUBLE, source, tag, comm, &status);

}

inline void MPI_Recv(GQTEN_Complex* data,
              const size_t data_size,
              const size_t source,
              const int tag,
              const MPI_Comm& comm){
  MPI_Status status;
  MPI_Recv( (void*)data, data_size, MPI_CXX_DOUBLE_COMPLEX, source, tag, comm, &status);

}

inline void MPI_Bcast(GQTEN_Double* data,
               const size_t data_size,
               const int root,
               const MPI_Comm& comm){
  MPI_Bcast((void*)data, data_size, MPI_DOUBLE, root, comm);
}

inline void MPI_Bcast(GQTEN_Complex * data,
               const size_t data_size,
               const int root,
               const MPI_Comm& comm){
  MPI_Bcast((void*)data, data_size, MPI_CXX_DOUBLE_COMPLEX, root, comm);
}


}//hp_numeric
}//gqten






#endif