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
    // Input, Bias and Output buffers
    void* i_gpu_ptr;
    void* l_gpu_ptr;
    void* o_gpu_ptr;
    try {
        // Allocate memory
        i_gpu_ptr = gcl_malloc(sizeof(cl_float) * input[0]->dims.Size(), input[0]->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        // Copy learnable parameters to GPU
        l_gpu_ptr = gcl_malloc(sizeof(cl_float) * learnable_params->dims.Size(), learnable_params->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        o_gpu_ptr = gcl_malloc(sizeof(cl_float) * output->dims.Size(), NULL, CL_MEM_WRITE_ONLY);
    } catch (...) {
        std::cerr << "Error allocating memory for BIAS FORWARDS PROPAGATION";
        throw InsufficientHardware();
    }
    
    // Execute bias addition
    dispatch_sync(*queue, ^{
        size_t glbl_size = input[0]->dims.Size();
        size_t wrk_grp_size = input[0]->dims.SizePerEx();
        size_t otpt_size = output->dims.Size();
        cl_ndrange range = {
            1,
            {0,0,0},
            {glbl_size,0,0},
            {NULL,0,0}
        };
        // Execute addition
        ElementWiseAdd_kernel(&range, (cl_float*)i_gpu_ptr, (cl_float*)o_gpu_ptr, (cl_float*)l_gpu_ptr);
        gcl_memcpy(output->data, o_gpu_ptr, otpt_size * sizeof(cl_float));
    });
    
    gcl_free(i_gpu_ptr);
    gcl_free(l_gpu_ptr);
    gcl_free(o_gpu_ptr);
}

void Layers::Bias::BackpropDeltas(Tensor** deltas, Tensor* backprop_deltas, Tensor* inp, LearnableParameters* learnable_params, void *params, dispatch_queue_t* queue) {
    // Deltas into bias layer are backpropagated straight through without changing. (multiplied by d/db(b)=1)
    memcpy(backprop_deltas->data, deltas[0]->data, sizeof(float) * deltas[0]->dims.Size());
}

void Layers::Bias::UpdateWeights(Tensor* deltas, Tensor* input, LearnableParameters* learnable_params, void* params, float eta, dispatch_queue_t* queue) {
    // Memory buffers
    void* d_gpu_ptr; // Deltas
    void* l_gpu_ptr; // Learnable parameters
    try {
        // Allocate memory
        d_gpu_ptr = gcl_malloc(sizeof(cl_float) * deltas->dims.Size(), deltas->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        l_gpu_ptr = gcl_malloc(sizeof(cl_float) * learnable_params->dims.Size(), learnable_params->data, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
    } catch (...) {
        std::cerr << "Error allocating memory for BIAS UPDATE WEIGHTS METHOD";
        throw InsufficientHardware();
    }
    
    // Execute bias addition
    dispatch_sync(*queue, ^{
        size_t glbl_size = deltas->dims.Size();
        size_t otpt_size = learnable_params->dims.Size();
        cl_ndrange range = {
            1,
            {0,0,0},
            {glbl_size,0,0},
            {NULL,0,0}
        };
        // Update bias weights
        UpdateBias_kernel(&range, (cl_float*)d_gpu_ptr, (cl_float*)l_gpu_ptr, eta);
        gcl_memcpy(learnable_params->data, l_gpu_ptr, otpt_size * sizeof(cl_float));
    });
    
    gcl_free(d_gpu_ptr);
    gcl_free(l_gpu_ptr);
}

// Initialise bias hyperparameters
Layers::Bias::Hyperparameters::Hyperparameters() {
    this->mean = 0;
    this->stddev = 0.1;
}

// Initialise hyperparameters
Layers::Bias::Hyperparameters::Hyperparameters(float mean, float stddev) {
    this->mean = mean;
    this->stddev = stddev;
}

// Initialise learnable parameters
LearnableParameters* Layers::Bias::InitialiseLearnableParameters(Layers::Bias::Hyperparameters p, Dims dims) {
    LearnableParameters* params = new LearnableParameters(dims);
    // Initialise using normal distribution
    params->InitialiseNormal(p.mean, p.stddev);
    return params;
}

Dims Layers::Bias::CalcOutputSize(Dims input) {
    return input;
}
