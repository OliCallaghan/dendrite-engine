//
//  FullyConnected.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef FullyConnected_hpp
#define FullyConnected_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "Tensor.hpp"
#include "LearnableParameters.hpp"
#include <OpenCL/opencl.h>

namespace Layers {
    namespace FullyConnected {
        void Forward(Tensor** input, Tensor* output, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void Backprop(Tensor** err, Tensor* backprop_err, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void UpdateWeights(Tensor* deriv, Tensor* input, LearnableParameters* learnable_params, void* params, float eta, dispatch_queue_t* queue);
        struct Hyperparameters {
            int Nodes;
            // Weight Params
            float mean;
            float stddev;
            Hyperparameters(int, float, float);
            Hyperparameters(int);
        };
        
        Dims CalcOutputSize(Dims input, Hyperparameters params);
        LearnableParameters* InitialiseLearnableParameters(Hyperparameters params, Dims dims);
        
        // Hyperparameters InitialiseLayer();
    }
}

#endif /* FullyConnected_hpp */
