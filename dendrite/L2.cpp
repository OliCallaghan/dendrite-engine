//
//  L2.cpp
//  dendrite
//
//  Created by Oli Callaghan on 23/11/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "L2.hpp"

// Calculates absolute loss value which is plotted on the loss graph
float Loss::L2::LossVal(Tensor* output, Tensor* prediction, dispatch_queue_t* queue) {
    // Memory buffers
    void* i_gpu_ptr = gcl_malloc(sizeof(cl_float) * output->dims.Size(), output->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* p_gpu_ptr = gcl_malloc(sizeof(cl_float) * prediction->dims.Size(), prediction->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* o_gpu_ptr = gcl_malloc(sizeof(cl_float), NULL, CL_MEM_WRITE_ONLY);
    
    float* rtrn = (float*)malloc(sizeof(float));
    float rtrn_single;
    
    // Calcualte loss
    dispatch_sync(*queue, ^{
        cl_ndrange range = {
            1,
            {0,0,0},
            {1,0,0},
            {1,0,0}
        };
        // Calculate loss
        L2_Val_Reduce_kernel(&range, (cl_float*)i_gpu_ptr, output->dims.Size(), (cl_float*)p_gpu_ptr, (cl_float*)o_gpu_ptr);
        gcl_memcpy(rtrn, o_gpu_ptr, sizeof(cl_float));
    });
    
    // Free memory on GPU
    gcl_free(i_gpu_ptr);
    gcl_free(p_gpu_ptr);
    gcl_free(o_gpu_ptr);
    
    rtrn_single = rtrn[0];
    free(rtrn);
    
    return rtrn_single;
}

void Loss::L2::Loss(Tensor* output, Tensor* prediction, Tensor* deltamap, dispatch_queue_t* queue) {
    // Memory buffers
    void* i_gpu_ptr = gcl_malloc(sizeof(cl_float) * output->dims.Size(), output->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* p_gpu_ptr = gcl_malloc(sizeof(cl_float) * prediction->dims.Size(), prediction->data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
    void* o_gpu_ptr = gcl_malloc(sizeof(cl_float) * output->dims.Size(), NULL, CL_MEM_WRITE_ONLY);
    
    // Calculate loss
    dispatch_sync(*queue, ^{
        size_t size = output->dims.Size();
        size_t size_ex = output->dims.SizePerEx();
        cl_ndrange range = {
            1,
            {0,0,0},
            {size_ex,0,0},
            {size,0,0}
        };
        // Calculate loss
        L2_kernel(&range, (cl_float*)i_gpu_ptr, (cl_float*)p_gpu_ptr, (cl_float*)o_gpu_ptr);
        gcl_memcpy(deltamap->data, o_gpu_ptr, sizeof(cl_float) * output->dims.Size());
    });
    
    // Free GPU memory
    gcl_free(i_gpu_ptr);
    gcl_free(p_gpu_ptr);
    gcl_free(o_gpu_ptr);
}
