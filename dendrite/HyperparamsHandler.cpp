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
        std::ofstream file;
        
        char match_pos = 1;
        int nodes;
        float mean, stddev;
        try {
            nodes = stoi(match[1]); match_pos++;
            mean = stof(match[2]); match_pos++;
            stddev = stof(match[4]); match_pos++;
        } catch (std::exception &e) {
            if (match_pos == 1) {
                throw ConversionError(match[match_pos], "Integer");
            } else {
                throw ConversionError(match[match_pos], "Float");
            }
        }
        
        Layers::FullyConnected::Hyperparameters hp(nodes, mean, stddev);
        
        try {
            // Save
            file.open(hp_loc.str(), std::ios::binary | std::ios::out);
            
            file.seekp(0);
            
            file.write((char*)(&hp), sizeof(hp));
            
            file.close();
        } catch (...) {
            throw FailedSavingHP(NULL);
        }
    }
}

void SaveB(std::string data, std::string loc, short n) {
    std::regex B_rgx("mean=(-?[0-9]+(.[0-9]+)?) stddev=(-?[0-9]+(.[0-9]+)?)");
    std::smatch match;
    
    std::stringstream hp_loc;
    hp_loc << loc << "hp" << n << ".dat";
    
    if (std::regex_match(data, match, B_rgx)) {
        std::ofstream file;
        
        char match_pos = 1;
        float mean, stddev;
        try {
            mean = stof(match[1]); match_pos++;
            stddev = stof(match[2]);
        } catch (std::exception &e) {
            throw ConversionError(match[match_pos], "Float");
        }
        
        Layers::Bias::Hyperparameters hp(mean, stddev);
        
        try {
            // Save
            file.open(hp_loc.str(), std::ios::binary | std::ios::out | std::ofstream::trunc);
            
            file.seekp(0);
            
            file.write((char*)(&hp), sizeof(hp));
            
            file.close();
        } catch (...) {
            throw FailedSavingHP(NULL);
        }
    }
}

void HyperparamsHandler::Save(std::string line) {
    size_t layer_t_pos, n_pos;
    std::string layer_t, line2, data;
    int n;
    try {
        layer_t_pos = line.find(" ");
        layer_t = line.substr(0, layer_t_pos);
        
        line2 = line.substr(layer_t_pos + 1); // Rest of line without layer_t
        
        n_pos = line2.find(" ");
        n = stoi(line2.substr(0, n_pos));
        
        data = line2.substr(n_pos + 1);
    } catch (...) {
        throw ;
    }
    
    
    if (layer_t == "FC") {
        SaveFC(data, this->loc, n);
    } else if (layer_t == "B") {
        SaveB(data, this->loc, n);
    }
}
