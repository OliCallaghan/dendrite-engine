
//
//  FullyConnected.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "FullyConnected.hpp"
#include "MatrixMultiply.cl.h"
#include <thread>
#include <iostream>
#include <vector>

// Forward propagation
void Layers::FullyConnected::Forward(Tensor** input, Tensor* output, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue) {
    // Input, Output and Weights buffers
    void* i_gpu_ptr;
    void* l_gpu_ptr;
    void* o_gpu_ptr;
    try {
        // Allocate memory
        i_gpu_ptr = gcl_malloc(sizeof(cl_float) * input[0]->dims.Size(), input[0]->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        l_gpu_ptr = gcl_malloc(sizeof(cl_float) * learnable_params->dims.Size(), learnable_params->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        o_gpu_ptr = gcl_malloc(sizeof(cl_float) * output->dims.Size(), NULL, CL_MEM_WRITE_ONLY);
    } catch (...) {
        std::cerr << "Error allocating memory for FC FORWARD PROPAGATION";
        throw InsufficientHardware();
    }
    
    size_t nodes = (*(Layers::FullyConnected::Hyperparameters*)params).Nodes;
    size_t output_size = nodes * input[0]->dims.dims[3];
    
    // Execute fully connected layer
    dispatch_sync(*queue, ^{
        size_t glbl_size = output_size; // Total size of output matrix
        size_t otpt_size = output->dims.Size(); // Should be same as glbl_size
        
        cl_ndrange range = {
            1,
            {0,0,0},
            {glbl_size,0,0},
            {NULL,0,0}
        };
        
        // Perform Matrix Multiplication
        MatrixMultiply_kernel(&range, (cl_float*)i_gpu_ptr, (cl_int)input[0]->dims.dims[0], (cl_float*)o_gpu_ptr, (cl_float*)l_gpu_ptr);
        gcl_memcpy(output->data, o_gpu_ptr, otpt_size * sizeof(cl_float));
    });
    
    // Free memory on GPU
    gcl_free(i_gpu_ptr);
    gcl_free(l_gpu_ptr);
    gcl_free(o_gpu_ptr);
}

void Layers::FullyConnected::Backprop(Tensor** err, Tensor* backprop_err, Tensor* inp, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue) {
    void* err_gpu_ptr;
    void* l_gpu_ptr;
    void* backproperr_gpu_ptr;
    try {
        err_gpu_ptr = gcl_malloc(sizeof(cl_float) * err[0]->dims.Size(), err[0]->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        l_gpu_ptr = gcl_malloc(sizeof(cl_float) * learnable_params->dims.Size(), learnable_params->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        backproperr_gpu_ptr = gcl_malloc(sizeof(cl_float) * backprop_err->dims.Size(), NULL, CL_MEM_WRITE_ONLY);
    } catch (...) {
        std::cerr << "Error allocating memory for FC BACKWARDS PROPAGATION";
        throw InsufficientHardware();
    }
    
    size_t nodes = learnable_params->dims.dims[0];
    size_t output_size = nodes * err[0]->dims.dims[3];
    
    // Execute fully connected layer backpropagation
    dispatch_sync(*queue, ^{
        size_t glbl_size = output_size; // Total size of output matrix
        size_t otpt_size = backprop_err->dims.Size(); // Should be same as glbl_size
        
        cl_ndrange range = {
            1,
            {0,0,0},
            {glbl_size,0,0},
            {NULL,0,0}
        };
        
        // Backpropagate deltas
        MatrixMultiplyTranspose_kernel(&range, (cl_float*)err_gpu_ptr, (cl_int)err[0]->dims.dims[0], (cl_float*)backproperr_gpu_ptr, (cl_float*)l_gpu_ptr);
        gcl_memcpy(backprop_err->data, backproperr_gpu_ptr, otpt_size * sizeof(cl_float));
    });
    
    // Free memory on GPU
    gcl_free(err_gpu_ptr);
    gcl_free(l_gpu_ptr);
    gcl_free(backproperr_gpu_ptr);
}

void Layers::FullyConnected::UpdateWeights(Tensor* deriv, Tensor* input, LearnableParameters* learnable_params, void* params, float eta, dispatch_queue_t* queue) {
    // Memory buffers
    void* deriv_gpu_ptr;
    void* l_gpu_ptr;
    void* wd_gpu_ptr;
    void* input_gpu_ptr;
    try {
        // Allocate memory
        deriv_gpu_ptr = gcl_malloc(sizeof(cl_float) * deriv->dims.Size(), deriv->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
        l_gpu_ptr = gcl_malloc(sizeof(cl_float) * learnable_params->dims.Size(), learnable_params->data, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
        wd_gpu_ptr = gcl_malloc(sizeof(cl_float) * learnable_params->dims.Size(), NULL, CL_MEM_READ_WRITE); // weight derivatives
        input_gpu_ptr = gcl_malloc(sizeof(cl_float) * input->dims.Size(), input->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    } catch (...) {
        std::cerr << "Error allocating memory for FC WEIGHT UPDATES METHOD";
        throw InsufficientHardware();
    }
    
    // Execute fully connected layer updates
    dispatch_sync(*queue, ^{
        size_t l_x = learnable_params->dims.dims[0];
        size_t l_y = learnable_params->dims.dims[1];
        
        size_t batch_size = input->dims.dims[3];
        
        cl_ndrange cwd_range = {
            2,
            {0,0,0},
            {l_x * batch_size,l_y * batch_size,0},
            // {l_x,l_y,0}
            {NULL,NULL,0}
        };
        
        cl_ndrange uw_range = {
            1,
            {0,0,0},
            {l_x * l_y * batch_size,0,0},
            // {l_x,l_y,0}
            {NULL,0,0}
        };
        
        // Calculate derivates with respect to each weight
        CalculateWeightDerivatives_kernel(&cwd_range, (cl_float*)deriv_gpu_ptr, (cl_float*)input_gpu_ptr, (cl_float*)wd_gpu_ptr);
        // Update each weight
        UpdateWeights_kernel(&uw_range, (cl_float*) wd_gpu_ptr, (cl_float*)l_gpu_ptr, eta);
        gcl_memcpy(learnable_params->data, l_gpu_ptr, l_x * l_y * sizeof(cl_float));
    });
    
    // Free memory on GPU
    gcl_free(deriv_gpu_ptr);
    gcl_free(l_gpu_ptr);
    gcl_free(wd_gpu_ptr);
    gcl_free(input_gpu_ptr);
}

// Initialise hyperparameters
Layers::FullyConnected::Hyperparameters::Hyperparameters(int Nodes) {
    this->Nodes = Nodes;
    this->mean = 0;
    this->stddev = 0.1;
}

// Initialise hyperparameters
Layers::FullyConnected::Hyperparameters::Hyperparameters(int Nodes, float mean, float stddev) {
    this->Nodes = Nodes;
    this->mean = mean;
    this->stddev = stddev;
}

Dims Layers::FullyConnected::CalcOutputSize(Dims input, Hyperparameters params) {
    Dims output({1,1,1,1});
    // Flattens to 1D array
    output.dims[0] = params.Nodes;
    output.dims[1] = 1;
    output.dims[2] = 1;
    output.dims[3] = input.dims[3];
    
    return output;
}

// Initialise learnable parameters
LearnableParameters* Layers::FullyConnected::InitialiseLearnableParameters(Layers::FullyConnected::Hyperparameters p, Dims dims) {
    LearnableParameters* params = new LearnableParameters(Dims({dims.dims[0], p.Nodes, 1, 1}));
    params->InitialiseNormal(p.mean, p.stddev);
    return params;
}
