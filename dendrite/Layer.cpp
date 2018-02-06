//
//  Layer.cpp
//  dendrite
//
//  Created by Oli Callaghan on 27/10/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "Layer.hpp"

size_t Layer::GetSizeOfHyperparameters() {
    switch (this->layer_t) {
        case Layers::Layer_T::FullyConnected_T:
            return sizeof(Layers::FullyConnected::Hyperparameters);
            break;
        case Layers::Layer_T::Bias_T:
            return sizeof(Layers::Bias::Hyperparameters);
            break;
        case Layers::Layer_T::Logistic_T:
            return 0;
            break;
        default:
            throw "Unknown Layer";
            break;
    }
}

Layer::Layer(Layers::Layer_T t, std::vector<short> i, std::vector<short> d, void* hyperparameters) {
    this->layer_t = t;
    this->input = i;
    this->dependents = d;
    this->hyperparameters = hyperparameters;
    
    // Set methods for layer
    switch (t) {
        case Layers::Layer_T::Bias_T:
            this->ForwardFunc = Layers::Bias::Forward;
            this->BackpropDeltasFunc = Layers::Bias::BackpropDeltas;
            this->CalcParamDeltasFunc = Layers::Bias::UpdateWeights;
            this->has_params = true;
            break;
        case Layers::Layer_T::FullyConnected_T:
            this->ForwardFunc = Layers::FullyConnected::Forward;
            this->BackpropDeltasFunc = Layers::FullyConnected::Backprop;
            this->CalcParamDeltasFunc = Layers::FullyConnected::UpdateWeights;
            this->has_params = true;
            break;
        case Layers::Layer_T::Logistic_T:
            this->ForwardFunc = Layers::Logistic::Forward;
            this->BackpropDeltasFunc = Layers::Logistic::Backprop;
            this->CalcParamDeltasFunc = Layers::Logistic::UpdateWeights;
            this->has_params = false;
        case Layers::Layer_T::Input_T:
            break;
        default:
            throw "Layer not implemented yet";
            break;
    }
}

void Layer::LoadLearnableParameters(std::string location, short id) {
    std::ifstream file;
    std::stringstream full_location;
    full_location << location << "lparams/lp" << id << ".dat";
    
    file.open(full_location.str(), std::ios::out | std::ios::binary);
    
    file.read((char*)(this->params->data), sizeof(float) * this->params->dims.Size());
}

bool Layer::SaveLearnableParameters(std::string location, short id) {
    std::ofstream file;
    std::stringstream full_location;
    full_location << location << "lparams/lp" << id << ".dat";
    
    file.open(full_location.str(), std::ios::out | std::ios::binary);
    file.seekp(0);
    
    // Output to File
    file.write((char*)(this->params->data), sizeof(float) * this->params->dims.Size());
    
    file.close();
    
    return true;
}

bool Layer::SaveHyperparameters(std::string location, short id) {
    std::ofstream file;
    std::stringstream full_location;
    full_location << location << "hparams/hp" << id << ".dat";
    
    file.open(full_location.str(), std::ios::out | std::ios::binary);
    file.seekp(0);
    
    file.write((char*)(this->hyperparameters), this->GetSizeOfHyperparameters());
    
    file.close();
    
    return true;
}
