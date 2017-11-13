//
//  Graph.cpp
//  dendrite
//
//  Created by Oli Callaghan on 26/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include <iostream>
#include <fstream>
#include "Graph.hpp"

Dims getDimsOfOutput(Dims* input, Layers::Layer_T layer_t, void* hyperparameters) {
    switch (layer_t) {
        case Layers::FullyConnected_T:
            return Layers::FullyConnected::CalcOutputSize(input, *(Layers::FullyConnected::Hyperparameters*)hyperparameters);
            break;
            
        default:
            throw "Unsupported Layer";
            break;
    }
    return Dims({1,1,1,1});
}

LearnableParameters* InitialiseLearnableParameters(Layers::Layer_T layer_t, void* hyperparameters) {
    switch (layer_t) {
        case Layers::FullyConnected_T:
            return Layers::FullyConnected::InitialiseLearnableParameters(*(Layers::FullyConnected::Hyperparameters*)hyperparameters);
            break;
            
        default:
            throw "Unsupported Layer";
            break;
    }
    return NULL;
}

bool Graph::Load() {
    std::ifstream model_struct;
    model_struct.open("model.struct");
    // Implement loading
    return true;
}

bool Graph::LoadFixed() {
    std::vector<short> input = {NULL};
    this->layers.push_back(*new Layer(Layers::Layer_T::Input_T, input, NULL));
    
    std::vector<short> i1 = {0};
    Layers::FullyConnected::Hyperparameters* h_p1 = new Layers::FullyConnected::Hyperparameters(3);
    this->layers.push_back(*new Layer(Layers::Layer_T::FullyConnected_T, i1, (void*)(h_p1)));
    
    std::vector<short> i2 = {1};
    Layers::FullyConnected::Hyperparameters* h_p2 = new Layers::FullyConnected::Hyperparameters(3);
    this->layers.push_back(*new Layer(Layers::Layer_T::FullyConnected_T, i2, (void*)(h_p2)));
    
    this->layer_n = this->layers.size();
    
    return true;
}

bool Graph::InitialiseLayers(Tensor* input) {
    // Check that layer 0 is input
    if (this->layers[0].layer_t != Layers::Layer_T::Input_T) {
        throw "ERR: First layer must be input layer!";
    }
    
    this->layers[0].output = input;
    
    for (int pos = 1; pos < this->layer_n; pos++) {
        std::cout << this->layers[0].output->dims->dims[0];
        this->layers[pos].output = new Tensor(getDimsOfOutput(this->layers[this->layers[pos].input[0]].output->dims, this->layers[pos].layer_t, this->layers[pos].hyperparameters));
        this->layers[pos].params = InitialiseLearnableParameters(this->layers[pos].layer_t, this->layers[pos].hyperparameters);
    }
    
    return true;
}

void Graph::Evaluate(Tensor* input, Tensor* output, dispatch_queue_t* queue) {
    for (int pos = 1; pos < this->layer_n; pos++) {
        // Iterate over every layer
        Tensor* inputs[this->layers[pos].input.size()];
        for (int inp = 0; inp < this->layers[pos].input.size(); inp++) {
            inputs[inp] = this->layers[this->layers[pos].input[inp]].output;
        }
        this->layers[pos].ForwardFunc(inputs, this->layers[pos].output, this->layers[pos].params, this->layers[pos].hyperparameters, queue);
    }
    
    output->data = this->layers[0].output->data;
}
