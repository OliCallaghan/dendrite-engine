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
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <functional>
#include <OpenCL/opencl.h>
#include "Layers.hpp"
#include "LearnableParameters.hpp"
#include "Exceptions.hpp"

// Layer Structure
struct Layer {
    Layers::Layer_T layer_t;
    std::vector<short> input; // Input index in layers array
    std::vector<short> dependents; // Input index in layers array
    LearnableParameters* params; // Learnable Parameters
    bool has_params;
    void* hyperparameters; // Hyperparameters
    
    size_t GetSizeOfHyperparameters();
    
    // Forward and backward propagation methods
    void (*ForwardFunc)(Tensor**, Tensor*, LearnableParameters*, void*, dispatch_queue_t*);
    void (*BackpropDeltasFunc)(Tensor**, Tensor*, Tensor*, LearnableParameters*, void*, dispatch_queue_t*);
    
    // Weight update methods
    void (*CalcParamDeltasFunc)(Tensor*, Tensor*, LearnableParameters*, void*, float, dispatch_queue_t*);
    
    // Methods that load learnable parameters form file
    void LoadLearnableParameters(std::string, short);
    bool SaveLearnableParameters(std::string, short);
    bool SaveHyperparameters(std::string, short);
    
    // Output and Delta buffers
    Tensor* output;
    Tensor* delta;
    
    // Initialiser
    Layer(Layers::Layer_T layer_t, std::vector<short> inputs, std::vector<short> dependents, void* hyperparameters);
};

#endif /* Layer_hpp */
