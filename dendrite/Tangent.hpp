//
//  Tangent.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Tangent_hpp
#define Tangent_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "Tensor.hpp"
#include "LearnableParameters.hpp"
#include <OpenCL/opencl.h>

namespace Layers {
    namespace Tanh {
        void Forward(Tensor** input, Tensor* output, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void Backprop(Tensor** err, Tensor* backprop_err, Tensor* inp, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void UpdateWeights(Tensor* deriv, Tensor* input, LearnableParameters* learnable_params, void* params, float eta, dispatch_queue_t* queue);
        struct Hyperparameters {
            
        };
        
        Dims CalcOutputSize(Dims input);
        LearnableParameters* InitialiseLearnableParameters(Hyperparameters params, Dims dims);
    }
}

#endif /* Tangent_hpp */
