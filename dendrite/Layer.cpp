//
//  Layer.cpp
//  dendrite
//
//  Created by Oli Callaghan on 27/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "Layer.hpp"

Layer::Layer(Layers::Layer_T t, std::vector<short> i, std::vector<short> d, void* hyperparameters) {
    this->layer_t = t;
    this->input = i;
    this->dependents = d;
    this->hyperparameters = hyperparameters;
    
    // Set methods for layer
    switch (t) {
        case Layers::Layer_T::Bias_T:
            this->ForwardFunc = Layers::Bias::Forward;
            break;
        case Layers::Layer_T::FullyConnected_T:
            this->ForwardFunc = Layers::FullyConnected::Forward;
            this->BackpropDeltasFunc = Layers::FullyConnected::Backprop;
            this->CalcParamDeltasFunc = Layers::FullyConnected::UpdateWeights;
            break;
        case Layers::Layer_T::Input_T:
            break;
        default:
            throw "Layer not implemented yet";
            break;
    }
}
