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
    
    //thread for contract, svd, qr
    inline unsigned tensor_manipulation_num_threads = kOmpDefaultNumThreads;
    
    inline void SetTensorManipulationThreads(unsigned thread){
        assert(thread>0);
        tensor_manipulation_num_threads = thread;
        mkl_set_num_threads_local( 0 );	
        mkl_set_num_threads(thread);
        mkl_set_dynamic(true);
    }
    //just for compitable. TODO: remove this API
    inline void SetTensorManipulationTotalThreads(unsigned thread){
        assert(thread>0);
        tensor_manipulation_num_threads = thread;
        mkl_set_num_threads_local( 0 );	
        mkl_set_num_threads(thread);
        mkl_set_dynamic(true);        
    }


    inline unsigned GetTensorManipulationThreads(){
        return tensor_manipulation_num_threads;
    }

} /* hp_numeric */
} /* gqten */



#endif /* ifndef GQTEN_FRAMEWORK_HP_NUMERIC_OMP_SET_H */
