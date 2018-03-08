//
//  Softmax.cpp
//  dendrite
//
//  Created by Oli Callaghan on 08/03/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#include "Softmax.hpp"
#include "Softmax.cl.h"
#include "Logistic.cl.h"

void Layers::Softmax::Forward(Tensor** input, Tensor* output, LearnableParameters* lparams, void* params, dispatch_queue_t* queue) {
    void* i_gpu_ptr;
    void* o_gpu_ptr;
    try {
        i_gpu_ptr = gcl_malloc(sizeof(cl_float) * input[0]->dims.Size(), input[0]->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        o_gpu_ptr = gcl_malloc(sizeof(cl_float) * output->dims.Size(), NULL, CL_MEM_WRITE_ONLY);
    } catch (...) {
        std::cerr << "Error allocating memory for SOFTMAX FORWARDS PROPAGATION";
        throw InsufficientHardware();
    }
    
    size_t output_size = output->dims.Size();
    
    // Execute bias addition
    dispatch_sync(*queue, ^{
        size_t glbl_size = output_size; // Total size of output matrix
        
        cl_ndrange range = {
            1,
            {0,0,0},
            {glbl_size,0,0},
            {NULL,0,0}
        };
        
        Softmax_kernel(&range, (cl_float*)i_gpu_ptr, (cl_float*)o_gpu_ptr);
        gcl_memcpy(output->data, o_gpu_ptr, output_size * sizeof(cl_float));
    });
    
    gcl_free(i_gpu_ptr);
    gcl_free(o_gpu_ptr);
}

void Layers::Softmax::Backprop(Tensor** err, Tensor* backprop_err, Tensor* inp, LearnableParameters* lparams, void* params, dispatch_queue_t* queue) {
    void* d_gpu_ptr;
    void* inpt_gpu_ptr;
    void* bd_gpu_ptr;
    
    try {
        d_gpu_ptr = gcl_malloc(sizeof(cl_float) * err[0]->dims.Size(), err[0]->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        inpt_gpu_ptr = gcl_malloc(sizeof(cl_float) * inp->dims.Size(), inp->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        bd_gpu_ptr = gcl_malloc(sizeof(cl_float) * backprop_err->dims.Size(), NULL, CL_MEM_WRITE_ONLY);
    } catch (...) {
        std::cerr << "Error allocating memory for SOFTMAX BACKWARDS PROPAGATION";
        throw InsufficientHardware();
    }
    
    size_t output_size = backprop_err->dims.Size();
    
    // Execute bias addition
    dispatch_sync(*queue, ^{
        size_t glbl_size = output_size; // Total size of output matrix
        
        cl_ndrange range = {
            1,
            {0,0,0},
            {glbl_size,0,0},
            {NULL,0,0}
        };
        
        Backprop_Softmax_kernel(&range, (cl_float*)inpt_gpu_ptr, (cl_float*)d_gpu_ptr, (cl_float*)bd_gpu_ptr);
        gcl_memcpy(backprop_err->data, bd_gpu_ptr, output_size * sizeof(cl_float));
    });
    
    gcl_free(d_gpu_ptr);
    gcl_free(inpt_gpu_ptr);
    gcl_free(bd_gpu_ptr);
}

void Layers::Softmax::UpdateWeights(Tensor*, Tensor*, LearnableParameters*, void*, float, dispatch_queue_t*) {
    // No weights to update
}

Dims Layers::Softmax::CalcOutputSize(Dims input) {
    return input;
}

LearnableParameters* Layers::Softmax::InitialiseLearnableParameters(Layers::Softmax::Hyperparameters hp, Dims dims) {
    return NULL;
}
