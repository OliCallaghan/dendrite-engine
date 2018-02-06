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
    } else if (t == "LG") {
        return Layers::Layer_T::Logistic_T;
    } else if (t == "B") {
        return Layers::Layer_T::Bias_T;
    } else {
        throw "Unsupported Layer Type";
    }
}

bool GraphLoader::ParseLayer(std::string line, GraphLoader::LayerDetails* layer) {
    std::regex layer_exprn("<lay (t)=([A-Z]+) (id)=([0-9]+) (i)=((?:[0-9]+)(?:,[0-9]+)*) (d)=((?:[0-9]+)(?:,[0-9]+)*)>");
    std::smatch match;
    
    Layers::Layer_T type;
    short id;
    std::vector<short> inpt = {};
    std::vector<short> dpt = {};
    
    if (std::regex_match(line, match, layer_exprn)) {
        // Match to layer
        for (unsigned int m = 0; m < match.size(); m++) {
            if (match[m] == "t") {
                // Layer type definition
                // Next match is type
                // match[m+1]
                
                // Determine layer type from match
                type = DetermineLayer(match[m+1]);
                
                // Skip next match
                m += 1;
            } else if (match[m] == "id") {
                // Layer ID definition
                id = std::stoi(match[m+1]);
                
                m += 1;
            } else if (match[m] == "i") {
                // Inputs definition
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
                // Stream inputs from input string to input array
                std::istringstream dpt_str(match[m+1]);
                std::string str_buf;
                while (getline(dpt_str, str_buf, ',')) {
                    dpt.push_back(std::stoi(str_buf));
                }
                
                m += 1;
            }
        }
        
        layer->type = type;
        layer->id = id;
        layer->inputs = inpt;
        layer->dependents = dpt;
        
        return true;
    } else {
        return false;
    }
}

Loss::Loss_T GraphLoader::ParseLoss(std::string line) {
    std::regex loss_exprn("<loss f=(L2)>");
    std::smatch match;
    if (std::regex_match(line, match, loss_exprn)) {
        // Parse loss function
        if (match[1] == "L2") {
            return Loss::Loss_T::L2_T;
        }
    }
    throw "Unsupported loss function";
}
