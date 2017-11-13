//
//  Bias.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "Bias.hpp"
#include "ElementWiseAdd.cl.h"

void Layers::Bias::Forward(Tensor** input, Tensor* output, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue) {
    // Maybe try and store learnable parameters on GPU?
    void* i_gpu_ptr = gcl_malloc(sizeof(cl_float) * input[0]->dims->Size(), input[0]->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* l_gpu_ptr = gcl_malloc(sizeof(cl_float) * learnable_params->dims->Size(), learnable_params->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* o_gpu_ptr = gcl_malloc(sizeof(cl_float) * output->dims->Size(), NULL, CL_MEM_WRITE_ONLY);
    
    // Execute bias addition
    dispatch_sync(*queue, ^{
        size_t glbl_size = input[0]->dims->Size();
        size_t wrk_grp_size = input[0]->dims->SizePerEx();
        cl_ndrange range = {
            1,
            {0,0,0},
            {glbl_size,0,0},
            {wrk_grp_size,0,0}
        };
        ElementWiseAdd_kernel(&range, (cl_float*)i_gpu_ptr, (cl_float*)o_gpu_ptr, (cl_float*)l_gpu_ptr);
        gcl_memcpy(output->data, o_gpu_ptr, glbl_size * sizeof(cl_float));
    });
    
    gcl_free(i_gpu_ptr);
    gcl_free(l_gpu_ptr);
    gcl_free(o_gpu_ptr);
}

Dims Layers::Bias::CalcOutputSize(Dims input) {
    return input;
}
