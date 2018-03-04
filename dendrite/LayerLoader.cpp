//
//  LayerLoader.cpp
//  dendrite
//
//  Created by Oli Callaghan on 12/12/2017.
//  Copyright © 2017 Oli Callaghan. All rights reserved.
//

#include "LayerLoader.hpp"
#include "Exceptions.hpp"

Layers::Layer_T DetermineLayer(std::string t) {
    if (t == "FC") {
        return Layers::Layer_T::FullyConnected_T;
    } else if (t == "LOG") {
        return Layers::Layer_T::Logistic_T;
    } else if (t == "B") {
        return Layers::Layer_T::Bias_T;
    } else {
        throw UnsupportedLayerType(t);
    }
}

bool GraphLoader::ParseLayer(std::string line, GraphLoader::LayerDetails* layer) {
    std::regex layer_exprn("<lay (t)=([A-Z]+) (id)=([0-9]+) (i)=((?:[0-9]+)(?:,[0-9]+)*) (d)=((?:[0-9]+)(?:,[0-9]+)*)>");
    std::smatch match;
    
    // set[] = {type, id, inpt, dpt}
    bool set[] = {false, false, false, false};
    
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
                layer->type = type;
                
                // Skip next match
                m += 1;
                set[0] = true;
            } else if (match[m] == "id") {
                // Layer ID definition
                id = std::stoi(match[m+1]);
                layer->id = id;
                
                m += 1;
                set[1] = true;
            } else if (match[m] == "i") {
                // Inputs definition
                // Stream inputs from input string to input array
                std::istringstream inpt_str(match[m+1]);
                std::string str_buf;
                while (getline(inpt_str, str_buf, ',')) {
                    inpt.push_back(std::stoi(str_buf));
                }
                
                layer->inputs = inpt;
                
                // Skip next match
                m += 1;
                set[2] = true;
            } else if (match[m] == "d") {
                // Dependents definition
                // Stream inputs from input string to input array
                std::istringstream dpt_str(match[m+1]);
                std::string str_buf;
                while (getline(dpt_str, str_buf, ',')) {
                    dpt.push_back(std::stoi(str_buf));
                }
                
                layer->dependents = dpt;
                
                m += 1;
                set[3] = true;
            }
        }
        
        if ((set[0] == false) || (set[1] == false) || (set[2] == false) || (set[3] == false)) {
            throw ModelStructSyntaxError(line, "<lay t=TYPE id=ID i=INPUTS d=DEPENDENTS");
        }
        
        return true;
    } else {
        return false;
    }
}

Loss::Loss_T GraphLoader::ParseLoss(std::string line) {
    std::regex loss_exprn("<loss f=([A-Za-z0-9]+)>");
    std::smatch match;
    if (std::regex_match(line, match, loss_exprn)) {
        // Parse loss function
        if (match[1] == "L2") {
            return Loss::Loss_T::L2_T;
        } else {
            throw UnsupportedLossFunction(match[1]);
        }
    } else {
        throw ModelStructSyntaxError(line, "<loss f=LOSS_FN>");
    }
}
