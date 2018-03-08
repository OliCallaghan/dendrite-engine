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
        case Layers::Layer_T::ReLU_T:
        case Layers::Layer_T::Softmax_T:
        case Layers::Layer_T::Tangent_T:
            return 0;
            break;
        default:
            throw UnsupportedLayerType("UNKNOWN");
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
            break;
        case Layers::Layer_T::ReLU_T:
            this->ForwardFunc = Layers::ReLU::Forward;
            this->BackpropDeltasFunc = Layers::ReLU::Backprop;
            this->CalcParamDeltasFunc = Layers::ReLU::UpdateWeights;
            this->has_params = false;
            break;
        case Layers::Layer_T::Softmax_T:
            this->ForwardFunc = Layers::Softmax::Forward;
            this->BackpropDeltasFunc = Layers::Softmax::Backprop;
            this->CalcParamDeltasFunc = Layers::Softmax::UpdateWeights;
            this->has_params = false;
            break;
        case Layers::Layer_T::Tangent_T:
            this->ForwardFunc = Layers::Tanh::Forward;
            this->BackpropDeltasFunc = Layers::Tanh::Backprop;
            this->CalcParamDeltasFunc = Layers::Tanh::UpdateWeights;
            this->has_params = false;
            break;
        case Layers::Layer_T::Input_T:
            break;
        default:
            throw UnsupportedLayerType("UNKNOWN");
            break;
    }
}

void Layer::LoadLearnableParameters(std::string location, short id) {
    std::ifstream file;
    std::stringstream full_location;
    full_location << location << "lparams/lp" << id << ".dat";
    
    try {
        file.open(full_location.str(), std::ios::ate | std::ios::out | std::ios::binary);
        if (file.fail()) {
            std::cerr << "File does not exist\n";
            throw FailedLoadingLP(id, "UNKNOWN");
        }
        
        size_t len = file.tellg();
        file.seekg(0);
        
        if ((this->params->dims.Size() * sizeof(float)) != len) {
            std::cerr << "Learnable params allocated size does not match file size\n";
            throw FailedLoadingLP(id, "UNKNOWN");
        }
        
        file.read((char*)this->params->data, sizeof(float) * this->params->dims.Size());
        file.close();
    } catch (FailedLoadingLP& e) {
        std::cout << this->params->dims.GetSizeStr() << "\n";
        
        file.close();
        std::cerr << e.what();
        //throw FailedLoadingLP(id, "UNKNOWN");
    }
}

bool Layer::SaveLearnableParameters(std::string location, short id) {
    std::ofstream file;
    std::stringstream full_location;
    full_location << location << "lparams/lp" << id << ".dat";
    
    try {
        file.open(full_location.str(), std::ios::out | std::ios::binary | std::ofstream::trunc);
        if (file.fail()) {
            std::cerr << "Error opening file";
            throw;
        }
        
        file.seekp(0);
        
        // Output to File
        file.write((char*)(this->params->data), sizeof(float) * this->params->dims.Size());
        
        file.close();
        
        return true;
    } catch (...) {
        file.close();
        throw FailedSavingLP(id);
    }
}

bool Layer::SaveHyperparameters(std::string location, short id) {
    std::ofstream file;
    std::stringstream full_location;
    full_location << location << "hparams/hp" << id << ".dat";
    
    try {
        file.open(full_location.str(), std::ios::out | std::ios::binary | std::ofstream::trunc);
        if (file.fail()) {
            std::cerr << "Error opening file";
            throw;
        }
        
        file.seekp(0);
        
        file.write((char*)(this->hyperparameters), this->GetSizeOfHyperparameters());
        
        file.close();
        
        return true;
    } catch (...) {
        file.close();
        throw FailedSavingHP(id);
    }
}
