//
//  Layer.hpp
//  dendrite
//
//  Created by Oli Callaghan on 27/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <stdio.h>
#include <vector>
#include <functional>
#include <OpenCL/opencl.h>
#include "Layers.hpp"
#include "LearnableParameters.hpp"

struct Layer {
    Layers::Layer_T layer_t;
    std::vector<short> input; // Input index in layers array
    std::vector<short> dependents; // Input index in layers array
    LearnableParameters* params;
    void* hyperparameters;
    
    // Forward and backward propagation methods
    std::function<void(Tensor**, Tensor*, LearnableParameters*, void*, dispatch_queue_t*)> ForwardFunc;
    std::function<void(Tensor**, Tensor*, LearnableParameters*, void*, dispatch_queue_t*)> BackpropDeltasFunc;
    std::function<void(Tensor*, Tensor*, LearnableParameters*, void*, float, dispatch_queue_t*)> CalcParamDeltasFunc;
    
    Tensor* output;
    Tensor* delta;
    
    // Initialiser
    Layer(Layers::Layer_T layer_t, std::vector<short> inputs, std::vector<short> dependents, void* hyperparameters);
};

#endif /* Layer_hpp */
