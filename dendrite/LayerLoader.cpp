//
//  LayerLoader.cpp
//  dendrite
//
//  Created by Oli Callaghan on 12/12/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "LayerLoader.hpp"

Layers::Layer_T DetermineLayer(std::string t) {
    if (t == "FC") {
        return Layers::Layer_T::FullyConnected_T;
    } else {
        throw "Unsupported Layer Type";
    }
}

Layer* GraphLoader::ParseLine(std::string line) {
    std::regex input_exprn("<inp s=((?:[0-9]+)(?:,[0-9]+){3})>");
    std::regex layer_exprn("<lay (t)=([A-Z]+) (id)=([0-9]+) (i)=((?:[0-9]+)(?:,[0-9]+)*) (d)=((?:[0-9]+)(?:,[0-9]+)*)>");
    std::smatch match;
    
    Layers::Layer_T type;
    std::vector<short> inpt = {};
    std::vector<short> dpt = {};
    void* hyperparams;
    
    if (std::regex_match(line, match, input_exprn)) {
        // Match to input layer
        // match[1] : Size of input
        std::cout << "INPUT LAYER: [ SIZE = {";
        
        // Define input layer parameters
        type = Layers::Layer_T::Input_T;
        inpt = {NULL};
        dpt = {NULL};
        hyperparams = NULL;
        
        // Logging (Debug)
        std::cout << match[1] << "} ]\n";
    } else if (std::regex_match(line, match, layer_exprn)) {
        // Match to layer
        std::cout << "NEW LAYER: [";
        for (unsigned int m = 0; m < match.size(); m++) {
            //std::cout << "[" << match[m] << "] ";
            if (match[m] == "t") {
                // Layer type definition
                // Next match is type
                // match[m+1]
                std::cout << " TYPE = " << match[m+1] << ",";
                
                // Determine layer type from match
                type = DetermineLayer(match[m+1]);
                
                // Skip next match
                m += 1;
            } else if (match[m] == "id") {
                // Layer ID definition
                std::cout << " ID = " << match[m+1] << ",";
                
                // Implement reordering
                
                m += 1;
            } else if (match[m] == "i") {
                // Inputs definition
                std::cout << " INPUTS = {" << match[m+1] << "},";
                
                // Stream inputs from input string to input array
                std::istringstream inpt_str(match[m+1]);
                std::string str_buf;
                while (getline(inpt_str, str_buf, ',')) {
                    inpt.push_back(std::stoi(str_buf));
                }
                
                // Skip next match
                m += 1;
            } else if (match[m] == "d") {
                // Dependents definition
                std::cout << " DEPENDENTS = {" << match[m+1] << "} ]\n";
                
                // Stream inputs from input string to input array
                std::istringstream dpt_str(match[m+1]);
                std::string str_buf;
                while (getline(dpt_str, str_buf, ',')) {
                    dpt.push_back(std::stoi(str_buf));
                }
                
                m += 1;
            }
        }
    }
    
    //return Layers::Layer_T::Bias_T;
    return new Layer(type, inpt, dpt, hyperparams);
}

Layer* ParseLoss(std::string line) {
    std::regex loss_exprn("<loss t=(L2)>");
    std::smatch match;
    if (std::regex_match(line, match, loss_exprn) {
        // Parse loss function
    }
}
