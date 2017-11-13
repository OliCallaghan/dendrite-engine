
//
//  FullyConnected.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "FullyConnected.hpp"
#include <thread>
#include <iostream>
#include <vector>

void Layers::FullyConnected::Forward(Tensor** input, Tensor* output, LearnableParameters* learnable_params, void* params, dispatch_queue_t* queue) {
    output->data = input[0]->data;
}

Layers::FullyConnected::Hyperparameters::Hyperparameters(int Nodes) {
    this->Nodes = Nodes;
    this->mean = 0;
    this->stddev = 0.01;
}

Layers::FullyConnected::Hyperparameters::Hyperparameters(int Nodes, float mean, float stddev) {
    this->Nodes = Nodes;
    this->mean = mean;
    this->stddev = stddev;
}

Dims Layers::FullyConnected::CalcOutputSize(Dims* input, Hyperparameters params) {
    Dims output({1,1,1,1});
    // std::cout << input;
    // Flattens to 1D array
    output.dims[0] = params.Nodes;
    output.dims[1] = 1;
    output.dims[2] = 1;
    output.dims[3] = (*input).dims[3];
    
    return output;
}

LearnableParameters* Layers::FullyConnected::InitialiseLearnableParameters(Layers::FullyConnected::Hyperparameters p) {
    LearnableParameters* params = new LearnableParameters(Dims({p.Nodes, 1, 1, 1}));
    return params;
}
