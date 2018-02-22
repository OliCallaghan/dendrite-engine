//
//  HyperparamsHandler.cpp
//  dendrite
//
//  Created by Oli Callaghan on 08/02/2018.
//  Copyright © 2018 Oli Callaghan. All rights reserved.
//

#include "HyperparamsHandler.hpp"

HyperparamsHandler::HyperparamsHandler(std::string loc) {
    this->loc = loc.append("hparams/");
}

void SaveFC(std::string data, std::string loc, short n) {
    std::regex FC_rgx("nodes=([0-9]+) mean=(-?[0-9]+(.[0-9]+)?) stddev=(-?[0-9]+(.[0-9]+)?)");
    std::smatch match;
    
    std::stringstream hp_loc;
    hp_loc << loc << "hp" << n << ".dat";
    
    if (std::regex_match(data, match, FC_rgx)) {
        std::ofstream file(hp_loc.str(), std::ios::binary | std::ios::out);
        
        int nodes = stoi(match[1]);
        float mean = stof(match[2]);
        float stddev = stof(match[4]);
        
        Layers::FullyConnected::Hyperparameters hp(nodes, mean, stddev);
        
        // Save
        file.seekp(0);
        
        file.write((char*)(&hp), sizeof(hp));
        
        file.close();
    }
}

void SaveB(std::string data, std::string loc, short n) {
    std::regex B_rgx("mean=(-?[0-9]+(.[0-9]+)?) stddev=(-?[0-9]+(.[0-9]+)?)");
    std::smatch match;
    
    std::stringstream hp_loc;
    hp_loc << loc << "hp" << n << ".dat";
    
    if (std::regex_match(data, match, B_rgx)) {
        std::ofstream file(hp_loc.str(), std::ios::binary | std::ios::out);
        
        float mean = stof(match[1]);
        float stddev = stof(match[2]);
        
        Layers::Bias::Hyperparameters hp(mean, stddev);
        
        // Save
        file.seekp(0);
        
        file.write((char*)(&hp), sizeof(hp));
        
        file.close();
    }
}

void HyperparamsHandler::Save(std::string line) {
    size_t layer_t_pos = line.find(" ");
    std::string layer_t = line.substr(0, layer_t_pos);
    
    std::string line2 = line.substr(layer_t_pos + 1); // Rest of line without layer_t
    
    size_t n_pos = line2.find(" ");
    int n = stoi(line2.substr(0, n_pos));
    
    std::string data = line2.substr(n_pos + 1);
    
    if (layer_t == "FC") {
        SaveFC(data, this->loc, n);
    } else if (layer_t == "B") {
        SaveB(data, this->loc, n);
    }
}
