//
//  LayerSaver.cpp
//  dendrite
//
//  Created by Oli Callaghan on 13/12/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "LayerSaver.hpp"

std::string LayerTypeStr(Layers::Layer_T t) {
    switch (t) {
        case Layers::FullyConnected_T:
            return "FC";
            break;
            
        default:
            throw "Unsupported Layer";
            break;
    }
}

std::string GetContents(std::vector<short> inputs) {
    std::string str;
    for (int i = 0; i < inputs.size(); i++) {
        str.append(std::to_string(inputs[i]));
    }
    return str;
}

bool GraphSaver::Save(std::string loc, std::vector<Layer> layers, Loss::Loss_T loss_t) {
    std::ofstream output;
    output.open(loc);
    
    // Output input layer
    output << "<inp s=" << layers[0].output->dims.GetSizeStr(',') << ">";
    
    for (int i = 1; i < layers.size(); i++) {
        output << "<lay t=" << LayerTypeStr(layers[i].layer_t) << " id=" << i << " i=" << GetContents(layers[i].input) << " d=" << GetContents(layers[i].dependents) << ">";
    }
    return true;
}
