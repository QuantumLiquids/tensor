// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2021-09-27 12:34
*
* Description: GraceQ/tensor project. MPI related
*/

/**
@file blas_level1.h
@brief Wrapper for the MPI related functions.
*/

#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_MPI_FUN_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_MPI_FUN_H

#include "gqten/framework/value_t.h"      // GQTEN_Double, GQTEN_Complex
#include "mpi.h"                          // MPI_Send, MPI_Recv...

namespace gqten {
/// High performance numerical functions.
namespace hp_numeric {
const size_t kAssumedMPICommunicationMaxDataLength = 4e9;      // char, multiples of 16 (size of complex number)
const size_t kAssumedMPICommunicationMaxDoubleDataSize = kAssumedMPICommunicationMaxDataLength / sizeof(GQTEN_Double);
const size_t kAssumedMPICommunicationMaxComplexDataSize = kAssumedMPICommunicationMaxDataLength / sizeof(GQTEN_Complex);

///< Send a solely number with type of size_t
inline void MPI_Send(const size_t n,
                     const int dest,
                     const int tag,
                     const MPI_Comm &comm) {
  ::MPI_Send((const void *) (&n), 1, MPI_UNSIGNED_LONG_LONG, dest, tag, comm);
}

inline void MPI_Recv(size_t &n,
                     const int source,
                     const int tag,
                     const MPI_Comm &comm) {
  ::MPI_Status status;
  ::MPI_Recv((void *) (&n), 1, MPI_UNSIGNED_LONG_LONG, source, tag, comm, &status);
}

///< Block point-to-point communication wrapper
///< The upperbound of size_t is much large than int that is used in MPI API.
inline void MPI_Send(const GQTEN_Double *data,
                     const size_t data_size,
                     const int dest,
                     const int tag,
                     const MPI_Comm &comm
) {
  if (data_size <= kAssumedMPICommunicationMaxDoubleDataSize) {
    ::MPI_Send((const void *) data, data_size, MPI_DOUBLE, dest, tag, comm);
  } else {
    size_t num_fragments = data_size / kAssumedMPICommunicationMaxDoubleDataSize + 1;
    for (size_t i = 0; i < num_fragments - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      ::MPI_Send((const void *) fragment_start,
                 kAssumedMPICommunicationMaxDoubleDataSize,
                 MPI_DOUBLE,
                 dest,
                 tag + i,
                 comm);
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxDoubleDataSize * (num_fragments - 1);
    char *fragment_start = (char *) data + (num_fragments - 1) * kAssumedMPICommunicationMaxDataLength;
    ::MPI_Send((const void *) fragment_start,
               remain_data_size,
               MPI_DOUBLE,
               dest,
               tag + num_fragments - 1,
               comm);
  }

}

inline void MPI_Send(const GQTEN_Complex *data,
                     const size_t data_size,
                     const int dest,
                     const int tag,
                     const MPI_Comm &comm
) {
  if (data_size <= kAssumedMPICommunicationMaxComplexDataSize) {
    ::MPI_Send((const void *) data, data_size, MPI_CXX_DOUBLE_COMPLEX, dest, tag, comm);
  } else {
    size_t num_fragment = data_size / kAssumedMPICommunicationMaxComplexDataSize + 1;
    for (size_t i = 0; i < num_fragment - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      ::MPI_Send((const void *) fragment_start,
                 kAssumedMPICommunicationMaxComplexDataSize,
                 MPI_CXX_DOUBLE_COMPLEX,
                 dest,
                 tag + i,
                 comm);
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxComplexDataSize * (num_fragment - 1);
    char *fragment_start = (char *) data + (num_fragment - 1) * kAssumedMPICommunicationMaxDataLength;
    ::MPI_Send((const void *) fragment_start,
               remain_data_size,
               MPI_CXX_DOUBLE_COMPLEX,
               dest,
               tag + num_fragment - 1,
               comm);
  }
}

// Assume source != mpi_any_source
inline void MPI_Recv(GQTEN_Double *data,
                     const size_t data_size,
                     const int source,
                     const int tag,
                     const MPI_Comm &comm
) {
  ::MPI_Status status;
  if (data_size <= kAssumedMPICommunicationMaxDoubleDataSize) {
    ::MPI_Recv((void *) data, data_size, MPI_DOUBLE, source, tag, comm, &status);
  } else {
    size_t num_fragment = data_size / kAssumedMPICommunicationMaxDoubleDataSize + 1;
    for (size_t i = 0; i < num_fragment - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      ::MPI_Recv((void *) fragment_start,
                 kAssumedMPICommunicationMaxDoubleDataSize,
                 MPI_DOUBLE,
                 source,
                 tag + i,
                 comm,
                 &status);
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxDoubleDataSize * (num_fragment - 1);
    char *fragment_start = (char *) data + (num_fragment - 1) * kAssumedMPICommunicationMaxDataLength;
    ::MPI_Recv((void *) fragment_start,
               remain_data_size,
               MPI_DOUBLE,
               source,
               tag + num_fragment - 1,
               comm,
               &status);
  }
}

///< note sizeof(GQTEN_Complex) = 16, while sizeof(MPI_CXX_DOUBLE_COMPLEX) = 8
inline void MPI_Recv(GQTEN_Complex *data,
                     const size_t data_size,
                     const size_t source,
                     const int tag,
                     const MPI_Comm &comm) {
  ::MPI_Status status;
  if (data_size <= kAssumedMPICommunicationMaxComplexDataSize) {
    ::MPI_Recv((void *) data, data_size, MPI_CXX_DOUBLE_COMPLEX, source, tag, comm, &status);
  } else {
    size_t num_fragment = data_size / kAssumedMPICommunicationMaxComplexDataSize + 1;
    for (size_t i = 0; i < num_fragment - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      ::MPI_Recv((void *) fragment_start,
                 kAssumedMPICommunicationMaxComplexDataSize,
                 MPI_CXX_DOUBLE_COMPLEX,
                 source,
                 tag + i,
                 comm,
                 &status);
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxComplexDataSize * (num_fragment - 1);
    char *fragment_start = (char *) data + (num_fragment - 1) * kAssumedMPICommunicationMaxDataLength;
    ::MPI_Recv((void *) fragment_start,
               remain_data_size,
               MPI_CXX_DOUBLE_COMPLEX,
               source,
               tag + num_fragment - 1,
               comm,
               &status);
  }

}

inline void MPI_Bcast(GQTEN_Double *data,
                      const size_t data_size,
                      const int root,
                      const MPI_Comm &comm) {
  if (data_size <= kAssumedMPICommunicationMaxDoubleDataSize) {
    ::MPI_Bcast((void *) data, data_size, MPI_DOUBLE, root, comm);
  } else {
    size_t times_of_sending = data_size / kAssumedMPICommunicationMaxDoubleDataSize + 1;
    for (size_t i = 0; i < times_of_sending - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      ::MPI_Bcast((void *) fragment_start,
                  kAssumedMPICommunicationMaxDoubleDataSize,
                  MPI_DOUBLE,
                  root,
                  comm);
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxDoubleDataSize * (times_of_sending - 1);
    char *fragment_start = (char *) data + (times_of_sending - 1) * kAssumedMPICommunicationMaxDataLength;
    ::MPI_Bcast((void *) fragment_start,
                remain_data_size,
                MPI_DOUBLE,
                root,
                comm);
  }
}

inline void MPI_Bcast(GQTEN_Complex *data,
                      const size_t data_size,
                      const int root,
                      const MPI_Comm &comm) {
  if (data_size <= kAssumedMPICommunicationMaxComplexDataSize) {
    ::MPI_Bcast((void *) data, data_size, MPI_CXX_DOUBLE_COMPLEX, root, comm);
  } else {
    size_t times_of_sending = data_size / kAssumedMPICommunicationMaxComplexDataSize + 1;
    for (size_t i = 0; i < times_of_sending - 1; i++) {
      char *fragment_start = (char *) data + i * kAssumedMPICommunicationMaxDataLength;
      ::MPI_Bcast((void *) fragment_start,
                  kAssumedMPICommunicationMaxComplexDataSize,
                  MPI_CXX_DOUBLE_COMPLEX,
                  root,
                  comm);
    }
    size_t remain_data_size =
        data_size - kAssumedMPICommunicationMaxComplexDataSize * (times_of_sending - 1);
    char *fragment_start = (char *) data + (times_of_sending - 1) * kAssumedMPICommunicationMaxDataLength;
    ::MPI_Bcast((void *) fragment_start,
                remain_data_size,
                MPI_CXX_DOUBLE_COMPLEX,
                root,
                comm);
  }
}

}//hp_numeric
}//gqten

#endif