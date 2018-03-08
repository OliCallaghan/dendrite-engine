//
//  Softmax.hpp
//  dendrite
//
//  Created by Oli Callaghan on 08/03/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#ifndef Softmax_hpp
#define Softmax_hpp

#include <stdio.h>
#include "Layer.hpp"
#include "Tensor.hpp"
#include "LearnableParameters.hpp"
#include <OpenCL/opencl.h>

namespace Layers {
    namespace Softmax {
        void Forward(Tensor** input, Tensor* output, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void Backprop(Tensor** err, Tensor* backprop_err, Tensor* inp, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void UpdateWeights(Tensor* deriv, Tensor* input, LearnableParameters* learnable_params, void* params, float eta, dispatch_queue_t* queue);
        struct Hyperparameters {
            
        };
        
        Dims CalcOutputSize(Dims input);
        LearnableParameters* InitialiseLearnableParameters(Hyperparameters params, Dims dims);
    }
}

#endif /* Softmax_hpp */
