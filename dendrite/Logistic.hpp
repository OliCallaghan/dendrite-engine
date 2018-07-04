//
//  Logistic.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Logistic_hpp
#define Logistic_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "Tensor.hpp"
#include "LearnableParameters.hpp"
#include <OpenCL/opencl.h>

// Logistic Methods
namespace Layers {
    namespace Logistic {
        void Forward(Tensor** input, Tensor* output, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void Backprop(Tensor** err, Tensor* backprop_err, Tensor* inp, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void UpdateWeights(Tensor* deriv, Tensor* input, LearnableParameters* learnable_params, void* params, float eta, dispatch_queue_t* queue);
        struct Hyperparameters {
            
        };
        
        Dims CalcOutputSize(Dims input);
        LearnableParameters* InitialiseLearnableParameters(Hyperparameters params, Dims dims);
    }
}

#endif /* Logistic_hpp */
