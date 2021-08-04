// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author:   Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
            Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-11-21 15:27
*
* Description: API for setting openmp thread
*/

/**
@file omp_set.h
@brief API for setting openmp thread
*/
#ifndef GQTEN_FRAMEWORK_HP_NUMERIC_OMP_SET_H
#define GQTEN_FRAMEWORK_HP_NUMERIC_OMP_SET_H

#include "assert.h"

namespace gqten {
/// High performance numerical functions.
namespace hp_numeric {
    const unsigned kOmpDefaultTotalNumThreads = 4;
    
    //thread for contract, svd, qr
    unsigned tensor_manipulation_total_num_threads = kOmpDefaultTotalNumThreads;
    
    //nested thread for svd,qr; if tensor_decomp_outer_parallel_num_threads==1, no nested thread
    unsigned tensor_decomp_outer_parallel_num_threads = 1;
    unsigned tensor_decomp_inner_parallel_num_threads = tensor_manipulation_total_num_threads/tensor_decomp_outer_parallel_num_threads;

    void SetTensorManipulationTotalThreads(unsigned thread){
        assert(thread>0);
        tensor_transpose_num_threads = thread;
        tensor_manipulation_total_num_threads = thread;
        tensor_decomp_inner_parallel_num_threads = tensor_manipulation_total_num_threads/tensor_decomp_outer_parallel_num_threads;
    }

    void SetTensorDecompOuterParallelThreads(unsigned thread){
        assert(thread>0);
        tensor_decomp_outer_parallel_num_threads = thread;
        tensor_decomp_inner_parallel_num_threads = tensor_manipulation_total_num_threads/tensor_decomp_outer_parallel_num_threads;
    }

    unsigned GetTensorManipulationTotalThreads(){
        return tensor_manipulation_total_num_threads;
    }

    unsigned GetTensorDecompOuterParallelThreads(){
        return tensor_decomp_outer_parallel_num_threads;
    }
    unsigned GetTensorDecompInnerParallelThreads(){
        return tensor_decomp_inner_parallel_num_threads;
    }

} /* hp_numeric */
} /* gqten */



#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_OMP_SET_H */
