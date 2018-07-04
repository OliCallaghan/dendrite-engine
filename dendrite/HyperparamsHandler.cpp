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

// Generating fully connected layer hyperparameters
void SaveFC(std::string data, std::string loc, short n) {
    std::regex FC_rgx("nodes=([0-9]+) mean=(-?[0-9]+(.[0-9]+)?) stddev=(-?[0-9]+(.[0-9]+)?)");
    std::smatch match;
    
    std::stringstream hp_loc;
    hp_loc << loc << "hp" << n << ".dat";
    
    // Match fully connected hyperparameters from input string
    if (std::regex_match(data, match, FC_rgx)) {
        std::ofstream file;
        
        char match_pos = 1;
        int nodes;
        float mean, stddev;
        try {
            // Extract parameters
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
        
        // Create hyperparameters object
        Layers::FullyConnected::Hyperparameters hp(nodes, mean, stddev);
        
        try {
            // Save object
            file.open(hp_loc.str(), std::ios::binary | std::ios::out);
            
            file.seekp(0);
            
            file.write((char*)(&hp), sizeof(hp));
            
            file.close();
        } catch (...) {
            // Failure saving hyperparameters
            throw FailedSavingHP(NULL);
        }
    }
}

// Generating bias layer hyperparameters
void SaveB(std::string data, std::string loc, short n) {
    std::regex B_rgx("mean=(-?[0-9]+(.[0-9]+)?) stddev=(-?[0-9]+(.[0-9]+)?)");
    std::smatch match;
    
    std::stringstream hp_loc;
    hp_loc << loc << "hp" << n << ".dat";
    
    // Match bias layer hyperparameters to string
    if (std::regex_match(data, match, B_rgx)) {
        std::ofstream file;
        
        char match_pos = 1;
        float mean, stddev;
        try {
            // Extract parameters
            mean = stof(match[1]); match_pos++;
            stddev = stof(match[2]);
        } catch (std::exception &e) {
            throw ConversionError(match[match_pos], "Float");
        }
        
        // Initialise hyperparameters object
        Layers::Bias::Hyperparameters hp(mean, stddev);
        
        try {
            // Save object
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
        // Extract layer type to save
        layer_t_pos = line.find(" ");
        layer_t = line.substr(0, layer_t_pos);
        
        line2 = line.substr(layer_t_pos + 1); // Rest of line without layer_t
        
        // Extract layer number
        n_pos = line2.find(" ");
        n = stoi(line2.substr(0, n_pos));
        
        data = line2.substr(n_pos + 1);
    } catch (...) {
        throw ;
    }
    
    // Generate correct hyerparameters for that layer
    if (layer_t == "FC") {
        SaveFC(data, this->loc, n);
    } else if (layer_t == "B") {
        SaveB(data, this->loc, n);
    }
}
