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
#include <string>

Dims getDimsOfOutput(Dims input, Layers::Layer_T layer_t, void* hyperparameters) {
    switch (layer_t) {
        case Layers::FullyConnected_T:
            return Layers::FullyConnected::CalcOutputSize(input, *(Layers::FullyConnected::Hyperparameters*)hyperparameters);
            break;
            
        case Layers::Bias_T:
            return Layers::Bias::CalcOutputSize(input);
            
        default:
            throw "Unsupported Layer";
            break;
    }
}

LearnableParameters* InitialiseLearnableParameters(Layers::Layer_T layer_t, void* hyperparameters, Dims dims) {
    switch (layer_t) {
        case Layers::FullyConnected_T:
            return Layers::FullyConnected::InitialiseLearnableParameters(*(Layers::FullyConnected::Hyperparameters*)hyperparameters, dims);
            break;
            
        case Layers::Bias_T:
            return Layers::Bias::InitialiseLearnableParameters(*(Layers::Bias::Hyperparameters*)hyperparameters, dims);
            
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
    std::vector<short> i0 = {NULL};
    std::vector<short> d0 = {1};
    this->layers.push_back(*new Layer(Layers::Layer_T::Input_T, i0, d0, NULL));
    
    std::vector<short> i1 = {0};
    std::vector<short> d1 = {1};
    Layers::FullyConnected::Hyperparameters* h_p1 = new Layers::FullyConnected::Hyperparameters(3);
    this->layers.push_back(*new Layer(Layers::Layer_T::FullyConnected_T, i1, d1, (void*)(h_p1)));
    
    std::vector<short> i2 = {1};
    std::vector<short> d2 = {2};
    Layers::FullyConnected::Hyperparameters* h_p2 = new Layers::FullyConnected::Hyperparameters(3);
    this->layers.push_back(*new Layer(Layers::Layer_T::FullyConnected_T, i2, d2, (void*)(h_p2)));
    
    //std::vector<short> i3 = {2};
    //std::vector<short> d3 = {NULL};
    //Layers::Bias::Hyperparameters* h_p3 = new Layers::Bias::Hyperparameters(0, 0.1);
    //this->layers.push_back(*new Layer(Layers::Layer_T::Bias_T, i3, d3, h_p3));
    
    this->layer_n = this->layers.size();
    this->loss_fn = new Loss::LossFn(Loss::L2_T);
    
    return true;
}

bool Graph::InitialiseLayers(Tensor* input) {
    // Check that layer 0 is input
    if (this->layers[0].layer_t != Layers::Layer_T::Input_T) {
        throw "ERR: First layer must be input layer!";
    }
    
    this->layers[0].output = input;
    
    for (int pos = 1; pos < this->layer_n; pos++) {
        // Initialise output buffer
        this->layers[pos].output = new Tensor(getDimsOfOutput(this->layers[this->layers[pos].input[0]].output->dims, this->layers[pos].layer_t, this->layers[pos].hyperparameters));
        // Initialise delta buffer
        this->layers[pos].delta = new Tensor(getDimsOfOutput(this->layers[this->layers[pos].input[0]].output->dims, this->layers[pos].layer_t, this->layers[pos].hyperparameters));
        // Initialise learnable parameters
        this->layers[pos].params = InitialiseLearnableParameters(this->layers[pos].layer_t, this->layers[pos].hyperparameters, this->layers[this->layers[pos].input[0]].output->dims);
    }
    
    return true;
}

void Graph::Evaluate(Tensor* input, Tensor* output, dispatch_queue_t* queue) {
    std::cout << "INPUT: " << input->GetDataStr() << "\n";
    for (size_t pos = 1; pos < this->layer_n; pos++) {
        // Iterate over every layer
        Tensor* inputs[this->layers[pos].input.size()];
        for (int inp = 0; inp < this->layers[pos].input.size(); inp++) {
            inputs[inp] = this->layers[this->layers[pos].input[inp]].output;
        }
        this->layers[pos].ForwardFunc(inputs, this->layers[pos].output, this->layers[pos].params, this->layers[pos].hyperparameters, queue);
        std::cout << "INTERMEDIATE: " << this->layers[pos].output->GetDataStr() << "\n";
    }
    
    output->data = this->layers[this->layer_n - 1].output->data;
}

float Graph::Learn(Tensor* input, Tensor* prediction, dispatch_queue_t* queue, float eta) {
    //std::cout << "INPUT: " << input->GetDataStr() << "\n";
    for (size_t pos = 1; pos < this->layer_n; pos++) {
        // Iterate over every layer
        Tensor* inputs[this->layers[pos].input.size()];
        for (int inp = 0; inp < this->layers[pos].input.size(); inp++) {
            inputs[inp] = this->layers[this->layers[pos].input[inp]].output;
        }
        this->layers[pos].ForwardFunc(inputs, this->layers[pos].output, this->layers[pos].params, this->layers[pos].hyperparameters, queue);
        std::string str = this->layers[pos].output->GetDataStr();
        //std::cout << "INTERMEDIATE: " << this->layers[pos].output->GetDataStr() << "\n";
    }
    
    float loss = this->loss_fn->Loss_Val(this->layers[this->layer_n - 1].output, prediction, queue);
    
    // Calculate loss for final layer
    this->loss_fn->Loss(this->layers[this->layer_n - 1].output, prediction, this->layers[this->layer_n - 1].delta, queue);
    
    for (size_t pos = this->layer_n - 1; pos > 1; pos--) {
        // Collect array of dependents
        Tensor* dependents[this->layers[pos - 1].dependents.size()];
        
        for (int dpt = 0; dpt < this->layers[pos - 1].dependents.size(); dpt++) {
            dependents[dpt] = this->layers[this->layers[pos].dependents[dpt]].delta;
        }
        
        this->layers[pos].BackpropDeltasFunc(dependents, this->layers[pos - 1].delta, this->layers[pos].params, this->layers[pos].hyperparameters, queue);
        
        this->layers[pos].CalcParamDeltasFunc(this->layers[pos].delta, this->layers[this->layers[pos].input[0]].output, this->layers[pos].params, this->layers[pos].hyperparameters, eta, queue);
    }
    
    return loss;
}
