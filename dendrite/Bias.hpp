//
//  Bias.hpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Bias_hpp
#define Bias_hpp

#include <stdio.h>
#include <iostream>
#include <OpenCL/opencl.h>
#include "LearnableParameters.hpp"
#include "Exceptions.hpp"

// Bias methods
namespace Layers {
    namespace Bias {
        void Forward(Tensor** input, Tensor* output, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void BackpropDeltas(Tensor** deltas, Tensor* backpropagated_deltas, Tensor* inp, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue);
        void UpdateWeights(Tensor* deltas, Tensor* input, LearnableParameters* learnable_parameters, void* params, float eta, dispatch_queue_t* queue);

        struct Hyperparameters {
            float mean; // Mean of normally distributed weights
            float stddev; // Standard deviation of noramlly distributed weights
            Hyperparameters(float, float);
            Hyperparameters();
        };
        
        LearnableParameters* InitialiseLearnableParameters(Layers::Bias::Hyperparameters p, Dims dims);
        Dims CalcOutputSize(Dims input);
    }
}

#endif /* Bias_hpp */
